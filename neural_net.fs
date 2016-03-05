open FSharp.NeuralNet.Network
open FSharp.NeuralNet.LoadData

[<EntryPoint>]
let main args =
  //let layer_sizes = [784;30;10]
  //let net = Network(layer_sizes)
  //printfn "%A" net
  let data = load_data "./data/mnist.txt"
  0
