namespace FSharp.NeuralNet

open MathNet.Numerics.LinearAlgebra
open System.Linq

module Network =

  let sigmoid z = 1.0 / (1.0 + exp -z)

  type Layer =
    { weights: Matrix<float>; biases: Vector<float> }
    member this.length = Vector.length this.biases
    member this.batchBiases(batchLength) =
      DenseMatrix.ofColumnSeq(Enumerable.Repeat(this.biases, batchLength))
    static member rand(prevSize, size) =
      { weights = DenseMatrix.randomStandard<float> size prevSize
        biases = DenseVector.randomStandard<float> size }

  type DataPoint =
    { inputVector: Vector<float>; expectedVector: Vector<float> }

  type Batch =
    { data: DataPoint list }
    member private this.matrixOf f = this.data |> Seq.map f |> DenseMatrix.ofColumnSeq
    member this.inputMatrix = this.matrixOf (fun x -> x.inputVector)
    member this.expectedMatrix = this.matrixOf (fun x -> x.expectedVector)

  type Activations =
    // Each column represents a data point, and each row a node in this layer.
    { weightedInputs: Matrix<float>; activations: Matrix<float> }
    static member inputLayer(a) =
      { weightedInputs = a; activations = a }
    static member withWeightedInputs(z) =
      { weightedInputs = z; activations = Matrix.map sigmoid z }
    static member zeros(rows, columns) =
      let m = DenseMatrix.zero rows columns
      { weightedInputs = m; activations = m }

  type Network(layerSizes: int list) =
    let numLayers = layerSizes.Length

    let randomLayers =
      let skipLast = Seq.take (numLayers - 1) layerSizes
      let skipFirst = Seq.skip 1 layerSizes
      Seq.zip skipLast skipFirst |> Seq.map Layer.rand |> Seq.toList

    let feedForwardBatch layers (inputMatrix : Matrix<double>) =
      let batchSize = Matrix.columnCount inputMatrix
      let inputActivations = Activations.inputLayer(inputMatrix)
      let layerActivations =
        layers
        |> List.map (fun (x:Layer) -> Activations.zeros(x.length, batchSize))
      let initialActivations = inputActivations :: layerActivations |> List.toArray
      let processLayer (activations : Activations []) i =
        let thisLayer = layers.Item(i-1)
        let previousActivations = activations.[i-1]
        let batchBiases = thisLayer.batchBiases(batchSize)
        let weightedInputs = thisLayer.weights * previousActivations.activations + batchBiases
        Array.set activations i (Activations.withWeightedInputs(weightedInputs))
        activations
      Seq.fold processLayer initialActivations [ 1 .. List.length layers ]

    member this.evaluate(inputMatrix) =
      feedForwardBatch randomLayers inputMatrix

    member this.gradientDescent(train, test, epochs, batchSize, learningRate) =

      let runBatch layers (batch : Batch, i) =
        let batchLayerActivations = feedForwardBatch layers batch.inputMatrix
        layers
        // TODO:
        // 1. Feed forward for the entire batch at once, to get activations at each layer and z's at each layer (for each data point).
        // Result is for each layer, an "activation" matrix of size (layerNodes, batchSize)
        //                        and a "z's" matrix of size (layerNodes, batchSize)
        //
        // 2. Then get the error in the last layer (for all data points in the batch at once)
        // Result is an "error in last layer" matrix of size (lastLayerNodes, batchSize)
        //
        // 3. Then go backwards through the layers and calculate the error in all layers (for all data points at once)
        // Result is for each layer except the last, a "error in layer" matrix of size (layerNodes, batchSize)
        //
        // 4. Then to get the error in weights... For each layer do (activations_l-1 * error_l).
        // Result is a matrix of size (lastLayerNodes, layerNodes) and is the SUM of the errors in the relevant weight, over all data points

      let runEpoch initialLayers i =
        printfn "Epoch %i" i

        let batches =
          train
          |> Shuffle.shuffleList
          |> Seq.chunkBySize batchSize
          |> Seq.map (fun x -> { data = Seq.toList x })
          |> Seq.toList

        printfn "%i batches" (List.length batches)

        // TODO: Evaluate after each epoch
        Seq.fold runBatch initialLayers (Seq.mapi (fun i x -> x,i) batches)

      Seq.fold runEpoch randomLayers [ 1 .. epochs ]
