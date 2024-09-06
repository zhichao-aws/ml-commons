package org.opensearch.ml.engine.analysis;

import static org.opensearch.ml.common.utils.StringUtils.gson;
import static org.opensearch.ml.engine.utils.FileUtils.calculateFileHash;
import static org.opensearch.ml.engine.utils.FileUtils.deleteFileQuietly;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Supplier;

import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.model.MLModelFormat;
import org.opensearch.ml.common.transport.register.MLRegisterModelInput;
import org.opensearch.ml.engine.MLEngine;
import org.opensearch.ml.engine.ModelHelper;
import org.opensearch.ml.engine.utils.ZipUtils;

import com.google.common.io.CharStreams;
import com.google.gson.stream.JsonReader;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.training.util.DownloadUtils;
import ai.djl.training.util.ProgressBar;
import lombok.extern.log4j.Log4j2;

@Log4j2
public class HFModelTokenizer extends Tokenizer {
    public static String NAME = "model_tokenizer";
    public static MLEngine mlEngine;
    public static ModelHelper modelHelper;

    private final CharTermAttribute termAtt;

    // payload分数属性
    private final PayloadAttribute payloadAtt;

    private HuggingFaceTokenizer tokenizer;

    private Encoding encoding;

    private int tokenIdx = 0;
    private int overflowingIdx = 0;

    public static void setMLComponent(MLEngine mlEngine, ModelHelper modelHelper) {
        HFModelTokenizer.mlEngine = mlEngine;
        HFModelTokenizer.modelHelper = modelHelper;
    }

    public static HuggingFaceTokenizer initializeHFTokenizer(String name) {
        return withDJLContext(() -> HuggingFaceTokenizer.newInstance(name));
    }

    public static HuggingFaceTokenizer initializeHFTokenizerFromConfigString(String configString) {
        return withDJLContext(() -> {
            InputStream inputStream = new ByteArrayInputStream(configString.getBytes(StandardCharsets.UTF_8));
            try {
                return HuggingFaceTokenizer.newInstance(inputStream, null);
            } catch (IOException e) {
                throw new IllegalArgumentException("Fail to create tokenizer. " + e.getMessage());
            }
        });
    }

    public static HuggingFaceTokenizer initializeHFTokenizerFromResources() {
        return withDJLContext(() -> {
            try {
                return HuggingFaceTokenizer.newInstance(HFModelTokenizer.class.getResourceAsStream("tokenizer.json"), null);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    public static HuggingFaceTokenizer initializeHFTokenizer(String modelName, String version, MLModelFormat modelFormat) {
        // todo: figure out the version incremental in ml
        return withDJLContext(() -> {
            final String taskId = UUID.randomUUID().toString();
            try {
                return AccessController.doPrivileged((PrivilegedExceptionAction<HuggingFaceTokenizer>) () -> {
                    Path registerModelPath = mlEngine.getRegisterModelPath(taskId, modelName, version);

                    String modelMetaListCachePath = registerModelPath.resolve("model_meta_list.json").toString();
                    String modelMetaListUrl = mlEngine.getPrebuiltModelMetaListPath();
                    DownloadUtils.download(modelMetaListUrl, modelMetaListCachePath, new ProgressBar());
                    List<?> modelMetaList;
                    try (JsonReader reader = new JsonReader(new FileReader(modelMetaListCachePath))) {
                        modelMetaList = gson.fromJson(reader, List.class);
                    }
                    MLRegisterModelInput.MLRegisterModelInputBuilder builder = MLRegisterModelInput.builder();
                    builder.modelName(modelName).version(version).modelFormat(modelFormat);
                    if (!modelHelper.isModelAllowed(builder.build(), modelMetaList)) {
                        throw new IllegalArgumentException(
                            "This model is not in the pre-trained model list, please check your parameters."
                        );
                    }

                    String modelConfigCachePath = registerModelPath.resolve("config.json").toString();
                    String configFileUrl = mlEngine.getPrebuiltModelConfigPath(modelName, version, modelFormat);
                    String modelZipFileUrl = mlEngine.getPrebuiltModelPath(modelName, version, modelFormat);
                    DownloadUtils.download(configFileUrl, modelConfigCachePath, new ProgressBar());

                    Map<?, ?> modelConfig;
                    try (JsonReader reader = new JsonReader(new FileReader(modelConfigCachePath))) {
                        modelConfig = gson.fromJson(reader, Map.class);
                    }

                    if (modelConfig == null) {
                        throw new IllegalArgumentException("model config not found");
                    }

                    FunctionName functionName = FunctionName
                        .from(
                            modelConfig.containsKey("function_name")
                                ? (String) modelConfig.get("function_name")
                                : (String) modelConfig.get("model_task_type")
                        );

                    if (functionName != FunctionName.SPARSE_TOKENIZE) {
                        throw new IllegalArgumentException("model is not sparse tokenize");
                    }

                    String modelContentZipCachePath = registerModelPath.resolve("content.zip").toString();
                    DownloadUtils.download(modelZipFileUrl, modelContentZipCachePath, new ProgressBar());
                    modelHelper.verifyModelZipFile(modelFormat, modelContentZipCachePath, modelName, functionName);

                    File modelZipFile = new File(modelContentZipCachePath);
                    log.debug("download model to file {}", modelZipFile.getAbsolutePath());
                    assert calculateFileHash(modelZipFile).equals(modelConfig.get(MLRegisterModelInput.MODEL_CONTENT_HASH_VALUE_FIELD));

                    ZipUtils.unzip(modelZipFile, registerModelPath);
                    return HuggingFaceTokenizer.newInstance(registerModelPath.resolve("tokenizer.json"));
                });
            } catch (PrivilegedActionException e) {
                throw new RuntimeException(e);
            } finally {
                deleteFileQuietly(mlEngine.getRegisterModelPath(taskId));
            }
        });
    }

    private static HuggingFaceTokenizer withDJLContext(Supplier<HuggingFaceTokenizer> tokenizerSupplier) {
        try {
            return AccessController.doPrivileged((PrivilegedExceptionAction<HuggingFaceTokenizer>) () -> {
                ClassLoader contextClassLoader = Thread.currentThread().getContextClassLoader();
                try {
                    System.setProperty("PYTORCH_PRECXX11", "true");
                    System.setProperty("PYTORCH_VERSION", "1.13.1");
                    System.setProperty("DJL_CACHE_DIR", mlEngine.getMlCachePath().toAbsolutePath().toString());
                    // DJL will read "/usr/java/packages/lib" if don't set "java.library.path". That will throw
                    // access denied exception
                    System.setProperty("java.library.path", mlEngine.getMlCachePath().toAbsolutePath().toString());
                    System.setProperty("ai.djl.pytorch.num_interop_threads", "1");
                    System.setProperty("ai.djl.pytorch.num_threads", "1");
                    Thread.currentThread().setContextClassLoader(ai.djl.Model.class.getClassLoader());

                    return tokenizerSupplier.get();
                } catch (Throwable e) {
                    throw new MLException("error", e);
                } finally {
                    Thread.currentThread().setContextClassLoader(contextClassLoader);
                }
            });
        } catch (PrivilegedActionException e) {
            throw new MLException("error", e);
        }
    }

    public HFModelTokenizer() {
        this(HFModelTokenizer.initializeHFTokenizer("bert-base-uncased"));
    }

    public HFModelTokenizer(HuggingFaceTokenizer tokenizer) {
        termAtt = addAttribute(CharTermAttribute.class);
        payloadAtt = addAttribute(PayloadAttribute.class);
        this.tokenizer = tokenizer;
    }

    @Override
    public void reset() throws IOException {
        super.reset();
        tokenIdx = 0;
        overflowingIdx = -1;
        String inputStr = CharStreams.toString(input);
        encoding = tokenizer.encode(inputStr, false, true);
    }

    @Override
    final public boolean incrementToken() throws IOException {
        // todo: 1. overflowing handle 2. max length of index.analyze.max_token_count 3. other attributes
        clearAttributes();
        Encoding curEncoding = encoding;

        while (tokenIdx < curEncoding.getTokens().length || overflowingIdx < encoding.getOverflowing().length) {
            if (tokenIdx >= curEncoding.getTokens().length) {
                tokenIdx = 0;
                overflowingIdx++;
                if (overflowingIdx < encoding.getOverflowing().length) {
                    curEncoding = encoding.getOverflowing()[overflowingIdx];
                }
                continue;
            }
            termAtt.append(curEncoding.getTokens()[tokenIdx]);
            tokenIdx++;
            return true;
        }

        return false;
        // int intBits = Float.floatToIntBits(10.0f);
        // payloadAtt.setPayload(
        // new BytesRef(new byte[] { (byte) (intBits >> 24), (byte) (intBits >> 16), (byte) (intBits >> 8), (byte) (intBits) })
        // );
    }
}
