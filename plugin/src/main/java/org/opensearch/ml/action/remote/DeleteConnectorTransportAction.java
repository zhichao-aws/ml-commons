/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.remote;

import static org.opensearch.ml.common.CommonValue.ML_CONNECTOR_INDEX;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.delete.DeleteRequest;
import org.opensearch.action.delete.DeleteResponse;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.transport.connector.MLConnectorDeleteAction;
import org.opensearch.ml.common.transport.connector.MLConnectorDeleteRequest;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

@Log4j2
public class DeleteConnectorTransportAction extends HandledTransportAction<ActionRequest, DeleteResponse> {

    Client client;
    NamedXContentRegistry xContentRegistry;

    @Inject
    public DeleteConnectorTransportAction(
        TransportService transportService,
        ActionFilters actionFilters,
        Client client,
        NamedXContentRegistry xContentRegistry
    ) {
        super(MLConnectorDeleteAction.NAME, transportService, actionFilters, MLConnectorDeleteRequest::new);
        this.client = client;
        this.xContentRegistry = xContentRegistry;
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<DeleteResponse> actionListener) {
        MLConnectorDeleteRequest mlConnectorDeleteRequest = MLConnectorDeleteRequest.fromActionRequest(request);
        String connectorId = mlConnectorDeleteRequest.getConnectorId();
        DeleteRequest deleteRequest = new DeleteRequest(ML_CONNECTOR_INDEX, connectorId);

        try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
            client.delete(deleteRequest, new ActionListener<DeleteResponse>() {
                @Override
                public void onResponse(DeleteResponse deleteResponse) {
                    log.info("Completed Delete Connector Request, connector id:{} deleted", connectorId);
                    actionListener.onResponse(deleteResponse);
                }

                @Override
                public void onFailure(Exception e) {
                    log.error("Failed to delete ML connector " + connectorId, e);
                    actionListener.onFailure(e);
                }
            });
        } catch (Exception e) {
            log.error("Failed to delete ML connector " + connectorId, e);
            actionListener.onFailure(e);
        }
    }
}
