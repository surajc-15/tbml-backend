import numpy as numpy
import flwr as flwr
from model import TBML_DetectionModel
class modelStrategy(flwr.server.strategy.FedAvg):
    def aggregate_fit(self, curr_round, results, failures):
        print(
            f"[SERVER] Round {curr_round}: received {len(results)} fit results, "
            f"failures={len(failures)}"
        )
        aggregated_weights,_ = super().aggregate_fit(curr_round,results,failures) #ignore aggregated metrics for nowS
        if aggregated_weights is None:
            print(f"[SERVER] Round {curr_round}: aggregation failed (no weights returned)")
        else:
            print(f"[SERVER] Round {curr_round}: aggregation completed successfully")

        if aggregated_weights is not None and curr_round==10:
            print(f"saving global weights for round {curr_round}")

            #convert flwr byte stream to numpy array

            aggregated_array = flwr.common.parameters_to_ndarrays(aggregated_weights)

            #save the numpy array to a file
            numpy.savez("global_model.npz",*aggregated_array)
            print("global weights saved successfully")

        return aggregated_weights,_


if __name__ == "__main__":
    print("Starting Flower server with custom strategy for 3 clients...")

    #initialize model on server side (this is just a placeholder, you can replace it with your actual model)
    model = TBML_DetectionModel(kyc_dim=3,swift_dim=1,trade_dim=3)

    #extract weights and convert to flwr parameters
    ndarrays = [val.cpu().numpy() for val in model.state_dict().values()]

    initial_parameters = flwr.common.ndarrays_to_parameters(ndarrays)




    strategy = modelStrategy(
        fraction_fit =1.0,
        min_fit_clients =3,
        min_available_clients =3,
        initial_parameters = initial_parameters
    )
    flwr.server.start_server(
    server_address = "0.0.0.0:8085",
    config = flwr.server.ServerConfig(num_rounds=10),
    strategy = strategy

)
