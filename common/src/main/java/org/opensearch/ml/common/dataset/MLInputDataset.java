/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.dataset;

import java.io.IOException;

import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.opensearch.ml.common.MLCommonsClassLoader;

@Getter
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
@RequiredArgsConstructor
public abstract class MLInputDataset implements Writeable {
    MLInputDataType inputDataType;

    @Override
    public void writeTo(StreamOutput streamOutput) throws IOException {
        streamOutput.writeEnum(this.inputDataType);
    }

    public static MLInputDataset fromStream(StreamInput in) throws IOException {
        MLInputDataType inputDataType = in.readEnum(MLInputDataType.class);
        return MLCommonsClassLoader.initMLInstance(inputDataType, in, StreamInput.class);
    }
}
