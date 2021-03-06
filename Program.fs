open FSharp.NeuralNet.Network
open FSharp.NeuralNet.LoadData
open FSharp.NeuralNet.Costs
open MathNet.Numerics.LinearAlgebra
open System.Drawing
open System.Drawing.Imaging
open System.IO

let numberRepresentedByVector vector =
    Vector.maxIndex vector

let getPredictionMatrix resultMatrix =
        Matrix.mapCols (fun _ col ->
            let max = numberRepresentedByVector col
            DenseVector.init 10 (fun x -> if x = max then 1.0 else 0.0)) resultMatrix

let evaluate expected actual =
    let predictions = getPredictionMatrix actual.activations
    let correct = int (Matrix.sum (expected.* predictions))
    printfn " (%i/10,000) " correct

let writeImage (vector:Vector<float>) (path:string) =
    let b = new Bitmap(28, 28)
    for x in [ 0 .. 27 ] do
        for y in [ 0 .. 27 ] do
            let pixelVal = 255 - int (vector.[28 * y + x] * 255.0)
            b.SetPixel(x, y, Color.FromArgb(255, pixelVal, pixelVal, pixelVal))
    b.Save(path, ImageFormat.Png)

let writeResult i (inputVector, expected, actual) =
    let dir = sprintf "./output/classified-as-%i/" actual
    Directory.CreateDirectory(dir) |> ignore
    let path =
        if expected = actual then
            Path.Combine(dir, sprintf "correct-%i.png" i)
        else
            Path.Combine(dir, sprintf "incorrect-expected-%i-%i.png" expected i)

    writeImage inputVector path

[<EntryPoint>]
let main args =
  printfn "Loading data..."
  let data = loadData "./data/mnist.txt" |> Seq.toList
  printfn "Data loaded."

  printfn "Generating network..."
  let layerSizes = [784;100;10]
  let net = Network(layerSizes, crossEntropy)
  printfn "Created network: %A" net

  // Now do the learning
  let epochs = 60
  let batchSize = 10
  let learningRate = 0.1
  let l2Reg = 5.0
  printfn "Running the neural network..."
  printfn "Layer sizes: %A" layerSizes
  printfn "Epochs: %i" epochs
  printfn "Batch size: %i" batchSize
  printfn "Learning rate: %f" learningRate
  printfn "L2 regularization parameter: %f" l2Reg

  let train = data |> Seq.take 50000 |> Seq.toList
  let test = data |> Seq.skip 60000 |> Seq.take 10000 |> Seq.toList

  printfn "Training data: %i items" (List.length train)
  printfn "Training data: %i items" (List.length test)

  let randomLayers = net.randomLayers
  printf "Epoch 0"
  let testBatch = { data = test }
  evaluate testBatch.expectedMatrix (net.evaluate(testBatch, randomLayers))

  let learnedLayers =
    net.gradientDescent(
      train,
      test,
      epochs,
      batchSize,
      learningRate,
      l2Reg,
      randomLayers,
      evaluate)

  // Output the images
  printfn "Outputting images..."
  let finalPredictions =
    (getPredictionMatrix (net.evaluate(testBatch, learnedLayers).activations)).EnumerateColumns()
    |> Seq.map numberRepresentedByVector
  let finalExpected =
    testBatch.expectedMatrix.EnumerateColumns()
    |> Seq.map numberRepresentedByVector
  let inputs = testBatch.inputMatrix.EnumerateColumns()
  Seq.zip3 inputs finalExpected finalPredictions |> Seq.iteri writeResult
  0
