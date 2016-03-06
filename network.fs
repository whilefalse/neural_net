namespace FSharp.NeuralNet

open MathNet.Numerics.LinearAlgebra
open System.Linq

module Network =

  let sigmoid z = 1.0 / (1.0 + exp -z)

  type Layer =
    { weights: Matrix<float>; biases: Vector<float> }
    member this.length = Vector.length this.biases
    member this.batchBiases(batchLength) =
      SparseMatrix.ofColumnSeq(Enumerable.Repeat(this.biases, batchLength))

  type DataPoint =
    { inputVector: Vector<float>; expectedVector: Vector<float> }

  type Batch =
    { data: DataPoint list }
    member private this.matrixOf f = this.data |> Seq.map f |> SparseMatrix.ofColumnSeq
    member this.inputMatrix = this.matrixOf (fun x -> x.inputVector)
    member this.expectedMatrix = this.matrixOf (fun x -> x.expectedVector)

  type BatchLayerActivations =
    // Each column represents a data point, and each row a node in this layer.
    { weightedInputs: Matrix<float>; activations: Matrix<float> }
    static member inputLayer(activations) =
      { weightedInputs = activations; activations = activations }
    static member withWeightedInputs(w) =
      { weightedInputs = w; activations = Matrix.map sigmoid w }
    static member zeros(rows, columns) =
      let m = SparseMatrix.zero rows columns
      { weightedInputs = m; activations = m }

  type Network(layer_sizes: int list) =
    let num_layers = layer_sizes.Length

    let random_layers =
      let make_layer (last_layer_size, layer_size) = {
        weights = DenseMatrix.randomStandard<float> layer_size last_layer_size
        biases = DenseVector.randomStandard<float> layer_size
      }
      let skip_last = Seq.take (num_layers - 1) layer_sizes
      let skip_first = Seq.skip 1 layer_sizes
      Seq.zip skip_last skip_first |> Seq.map make_layer |> Seq.toList

    let feedForwardBatch layers (inputMatrix : Matrix<double>) =
      let batchSize = Matrix.columnCount inputMatrix
      let inputActivations = BatchLayerActivations.inputLayer(inputMatrix)
      let layerActivations =
        layers
        |> List.map (fun (x:Layer) -> BatchLayerActivations.zeros(x.length, batchSize))
      let initialActivations = inputActivations :: layerActivations |> List.toArray
      let processLayer (activations : BatchLayerActivations []) i =
        let thisLayer = layers.Item(i-1)
        let previousActivations = activations.[i-1]
        let batchBiases = thisLayer.batchBiases(batchSize)
        let weightedInputs = thisLayer.weights * previousActivations.activations + batchBiases
        Array.set activations i (BatchLayerActivations.withWeightedInputs(weightedInputs))
        activations
      Seq.fold processLayer initialActivations (seq { 1 .. List.length layers })

    member this.evaluate(inputMatrix) =
      feedForwardBatch random_layers inputMatrix

    member this.gradientDescent(training_data, test_data, numEpochs, batchSize, learning_rate) =
      let run_batch layers (batch : Batch, i) =
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

      let run_epoch epoch_initial_layers i =
        printfn "Epoch %i" i

        let batches =
          training_data
          |> Shuffle.shuffleList
          |> Seq.chunkBySize batchSize
          |> Seq.map (fun x -> { data = Seq.toList x })
          |> Seq.toList

        printfn "%i batches" (List.length batches)

        // TODO: Evaluate after each epoch
        Seq.fold run_batch epoch_initial_layers (Seq.mapi (fun i x -> x,i) batches)

      let epochs = seq { 1 .. numEpochs }
      Seq.fold run_epoch random_layers epochs
