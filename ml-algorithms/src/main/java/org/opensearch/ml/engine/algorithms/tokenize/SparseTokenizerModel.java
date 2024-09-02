/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.tokenize;

import static org.opensearch.ml.common.CommonValue.ML_MAP_RESPONSE_KEY;
import static org.opensearch.ml.common.utils.StringUtils.gson;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.dataset.MLInputDataset;
import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.parameter.sparsetokenize.SparseTokenizeParams;
import org.opensearch.ml.common.model.MLModelConfig;
import org.opensearch.ml.common.model.MLModelFormat;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.common.output.model.ModelResultFilter;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.utils.StringUtils;
import org.opensearch.ml.engine.Trainable;
import org.opensearch.ml.engine.algorithms.DLModel;
import org.opensearch.ml.engine.annotation.Function;
import org.opensearch.ml.engine.utils.ModelSerDeSer;

import com.google.gson.reflect.TypeToken;

import ai.djl.MalformedModelException;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import lombok.extern.log4j.Log4j2;

/**
 * Tokenizer model will load from two file: tokenizer file and IDF file.
 * For IDF file, it is a predefined weight for each Token. It is calculated like the BM25 IDF for each dataset.
 * Decouple idf with model inference will give a more tight upperbound for predicted token weight. And this can help accelerate the WAND algorithm for lucene 9.7.
 * IDF introduces global token information, boost search relevance.
 * In our pretrained Tokenizer model, we will provide a general IDF from MSMARCO. Customer could recalculate a IDF in their own dataset.
 * If without IDF, the weight of each token in the result will be set to 1.0.
 * Since we regard tokenizer as a model. Cusotmer needs to keep the consistency between tokenizer/model by themselves.
 */
@Log4j2
@Function(FunctionName.SPARSE_TOKENIZE)
public class SparseTokenizerModel extends DLModel implements Trainable {
    public static final String VERSION = "1.0.0";
    private HuggingFaceTokenizer tokenizer;

    private Map<String, Float> idf;

    public String IDF_FILE_NAME = "idf.json";

    @Override
    public ModelTensorOutput predict(String modelId, MLInput mlInput) throws TranslateException {
        MLInputDataset inputDataSet = mlInput.getInputDataset();
        List<ModelTensors> tensorOutputs = new ArrayList<>();
        TextDocsInputDataSet textDocsInput = (TextDocsInputDataSet) inputDataSet;
        ModelResultFilter resultFilter = textDocsInput.getResultFilter();
        for (String doc : textDocsInput.getDocs()) {
            Output output = new Output(200, "OK");
            Encoding encodings = tokenizer.encode(doc);
            long[] indices = encodings.getIds();
            List<ModelTensor> outputs = new ArrayList<>();
            String[] tokens = Arrays
                .stream(indices)
                .distinct()
                .mapToObj(value -> new long[] { value })
                .map(value -> this.tokenizer.decode(value, true))
                .filter(s -> !s.isEmpty())
                .toArray(String[]::new);
            Map<String, Float> tokenWeights = Arrays
                .stream(tokens)
                .collect(Collectors.toMap(token -> token, token -> idf.getOrDefault(token, 1.0f)));
            Map<String, ?> wrappedMap = Map.of(ML_MAP_RESPONSE_KEY, Collections.singletonList(tokenWeights));
            ModelTensor tensor = ModelTensor.builder().dataAsMap(wrappedMap).build();
            outputs.add(tensor);
            ModelTensors modelTensorOutput = new ModelTensors(outputs);
            output.add(modelTensorOutput.toBytes());
            tensorOutputs.add(parseModelTensorOutput(output, resultFilter));
        }
        return new ModelTensorOutput(tensorOutputs);
    }

    protected void doLoadModel(
        List<Predictor<Input, Output>> predictorList,
        List<ZooModel<Input, Output>> modelList,
        String engine,
        Path modelPath,
        MLModelConfig modelConfig
    ) throws ModelNotFoundException,
        MalformedModelException,
        IOException,
        TranslateException {
        tokenizer = HuggingFaceTokenizer.builder().optPadding(true).optTokenizerPath(modelPath.resolve("tokenizer.json")).build();
        idf = new HashMap<>();
        if (Files.exists(modelPath.resolve(IDF_FILE_NAME))) {
            Type mapType = new TypeToken<Map<String, Float>>() {
            }.getType();
            idf = gson.fromJson(new InputStreamReader(Files.newInputStream(modelPath.resolve(IDF_FILE_NAME))), mapType);
        }
        log.info("sparse tokenize Model {} is successfully deployed", modelId);
    }

    @Override
    public boolean isModelReady() {
        if (modelHelper == null || modelId == null || tokenizer == null) {
            return false;
        }
        return true;
    }

    @Override
    public void close() {
        if (modelHelper != null && modelId != null) {
            modelHelper.deleteFileCache(modelId);
            if (idf != null || tokenizer != null) {
                tokenizer = null;
                idf = null;
            }
        }
    }

    @Override
    public Translator<Input, Output> getTranslator(String engine, MLModelConfig modelConfig) {
        return null;
    }

    @Override
    public TranslatorFactory getTranslatorFactory(String engine, MLModelConfig modelConfig) {
        return null;
    }

    @Override
    public MLModel train(MLInput mlInput) {

        SparseTokenizeParams params = (SparseTokenizeParams) mlInput.getParameters();
        if (Objects.isNull(params) || !params.hasContent()) {
            throw new IllegalArgumentException("No tokenizer config information.");
        }

        String inputString = StringUtils.toJson(params.getTokenizerConfig());
//        InputStream inputStream = new ByteArrayInputStream(inputString.getBytes(StandardCharsets.UTF_8));
//        try {
//            tokenizer = HuggingFaceTokenizer.newInstance(inputStream, null);
//        } catch (IOException e) {
//            throw new IllegalArgumentException("Fail to create tokenizer. " + e.getMessage());
//        }

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ZipOutputStream zos = new ZipOutputStream(baos);

        // Adding the first JSON file
        ZipEntry entry1 = new ZipEntry("tokenizer.json");
        try {
            zos.putNextEntry(entry1);
            zos.write(inputString.getBytes());
            zos.closeEntry();
            zos.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        byte[] zipBytes = baos.toByteArray();

        // Now `zipBytes` contains the entire zip file as a byte array
        System.out.println("Zip array length: " + zipBytes.length);
        Instant now = Instant.now();

        MLModel mlModel = MLModel
                .builder()
                .modelId(modelId)
                .name(FunctionName.SPARSE_TOKENIZE.name())
                .algorithm(FunctionName.SPARSE_TOKENIZE)
                .version(VERSION)
                .modelFormat(MLModelFormat.TORCH_SCRIPT)
                .chunkNumber(0)
                .totalChunks(1)
                .content(Base64.getEncoder().encodeToString(zipBytes))
                .createdTime(now)
                .lastUpdateTime(now)
                .build();

        return mlModel;
    }
}
