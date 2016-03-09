open FSharp.NeuralNet.Network
open FSharp.NeuralNet.LoadData
open MathNet.Numerics.LinearAlgebra

[<EntryPoint>]
let main args =
  printfn "Loading data..."
  let data = loadData "./data/mnist.txt" |> Seq.toList
  printfn "Data loaded."

  printfn "Generating network..."
  let layerSizes = [784;30;10]
  let net = Network(layerSizes)
  printfn "Created network: %A" net

  // Now do the learning
  let epochs = 10
  let batchSize = 10
  let learningRate = 3.0
  printfn "Running the neural network..."
  printfn "Epochs: %i" epochs
  printfn "Batch size: %i" batchSize
  printfn "Learning rate: %f" learningRate

  let train = data |> Seq.take 50000 |> Seq.toList
  let test = data |> Seq.skip 60000 |> Seq.take 10000 |> Seq.toList

  printfn "Training data: %i items" (List.length train)
  printfn "Training data: %i items" (List.length test)

  printfn "Epoch 0"
  printfn "%i/10000" (net.evaluate(test))
  let learnedLayers =
    net.gradientDescent(
      train,
      test,
      epochs,
      batchSize,
      learningRate)
  0
