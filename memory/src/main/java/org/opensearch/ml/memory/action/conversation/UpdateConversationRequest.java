/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.memory.action.conversation;

import static org.opensearch.action.ValidateActions.addValidationError;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Map;

import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.core.common.io.stream.InputStreamStreamInput;
import org.opensearch.core.common.io.stream.OutputStreamStreamOutput;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentParser;

import lombok.Builder;
import lombok.Getter;

@Getter
public class UpdateConversationRequest extends ActionRequest {
    String conversationId;
    Map<String, Object> updateContent;

    @Builder
    public UpdateConversationRequest(String conversationId, Map<String, Object> updateContent) {
        this.conversationId = conversationId;
        this.updateContent = updateContent;
    }

    public UpdateConversationRequest(StreamInput in) throws IOException {
        super(in);
        this.conversationId = in.readString();
        this.updateContent = in.readMap();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(this.conversationId);
        out.writeMap(this.getUpdateContent());
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException exception = null;

        if (this.conversationId == null) {
            exception = addValidationError("conversation id can't be null", exception);
        }

        return exception;
    }

    public static UpdateConversationRequest parse(XContentParser parser, String conversationId) throws IOException {
        Map<String, Object> dataAsMap = null;
        dataAsMap = parser.map();

        return UpdateConversationRequest.builder().conversationId(conversationId).updateContent(dataAsMap).build();
    }

    public static UpdateConversationRequest fromActionRequest(ActionRequest actionRequest) {
        if (actionRequest instanceof UpdateConversationRequest) {
            return (UpdateConversationRequest) actionRequest;
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream(); OutputStreamStreamOutput osso = new OutputStreamStreamOutput(baos)) {
            actionRequest.writeTo(osso);
            try (StreamInput input = new InputStreamStreamInput(new ByteArrayInputStream(baos.toByteArray()))) {
                return new UpdateConversationRequest(input);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("failed to parse ActionRequest into UpdateConversationRequest", e);
        }
    }
}
