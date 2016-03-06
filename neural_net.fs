open FSharp.NeuralNet.Network
open FSharp.NeuralNet.LoadData
open MathNet.Numerics.LinearAlgebra

[<EntryPoint>]
let main args =
  printfn "Loading data..."
  let data = load_data "./data/mnist.txt" |> Seq.toList
  printfn "Data loaded."

  printfn "Generating network..."
  let layer_sizes = [784;30;10]
  let net = Network(layer_sizes)
  printfn "Created network: %A" net

  printfn "Guessing first 2 data points..."
  let inputMatrix =
    data
    |> Seq.take 2
    |> Seq.map (fun x -> x.inputVector)
    |> SparseMatrix.ofColumnSeq
  let expectedMatrix =
    data
    |> Seq.take 2
    |> Seq.map (fun x -> x.expectedVector)
    |> SparseMatrix.ofColumnSeq
  let outputMatrix = net.evaluate(inputMatrix)
  printfn "Expected output matrix: %A" expectedMatrix
  printfn "Actual output matrix: %A" outputMatrix

  // Now do the learning
  let epochs = 30
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

  let learnedLayers =
    net.gradientDescent(
      train,
      test,
      epochs,
      batchSize,
      learningRate)
  0
