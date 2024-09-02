package org.opensearch.ml.common.input.parameter.sparsetokenize;

import static org.opensearch.core.xcontent.XContentParserUtils.ensureExpectedToken;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;

import org.opensearch.core.ParseField;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.annotation.MLAlgoParameter;
import org.opensearch.ml.common.input.parameter.MLAlgoParams;

import lombok.Builder;
import lombok.Data;
import lombok.Getter;

@Data
@MLAlgoParameter(algorithms = { FunctionName.SPARSE_TOKENIZE })
@Getter
public class SparseTokenizeParams implements MLAlgoParams {
    public static final String PARSE_FIELD_NAME = FunctionName.SPARSE_TOKENIZE.name();
    public static final NamedXContentRegistry.Entry XCONTENT_REGISTRY = new NamedXContentRegistry.Entry(
        MLAlgoParams.class,
        new ParseField(PARSE_FIELD_NAME),
        it -> parse(it)
    );

    public static final String TOKENIZER_CONFIG_FIELD = "tokenizer_config";

    private Map tokenizerConfig;

    @Builder(toBuilder = true)
    public SparseTokenizeParams(Map tokenizerConfig) {
        this.tokenizerConfig = tokenizerConfig;
    }

    public SparseTokenizeParams(StreamInput in) throws IOException {
        if (in.readBoolean()) {
            this.tokenizerConfig = in.readMap();
        }
    }

    public static MLAlgoParams parse(XContentParser parser) throws IOException {
        Map tokenizerConfig = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case TOKENIZER_CONFIG_FIELD:
                    tokenizerConfig = parser.map();
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return new SparseTokenizeParams(tokenizerConfig);
    }

    @Override
    public String getWriteableName() {
        return PARSE_FIELD_NAME;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        if (Objects.nonNull(tokenizerConfig)) {
            out.writeBoolean(true);
            out.writeMap(tokenizerConfig);
        } else {
            out.writeBoolean(false);
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, ToXContent.Params params) throws IOException {
        builder.startObject();
        if (Objects.nonNull(tokenizerConfig)) {
            builder.field(TOKENIZER_CONFIG_FIELD, tokenizerConfig);
        }
        builder.endObject();
        return builder;
    }

    @Override
    public int getVersion() {
        return 1;
    }

    public boolean hasContent() {
        return Objects.nonNull(tokenizerConfig);
    }
}
