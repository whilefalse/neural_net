namespace FSharp.NeuralNet

open MathNet.Numerics.LinearAlgebra
open System.Linq

module Network =
  let sigmoid z = 1.0 / (1.0 + exp -z)
  let sigmoid' z = sigmoid(z) * (1.0 - sigmoid(z))
  let feedForward w a b = w * a + b
  let backPropErrors delta zs = delta.* (Matrix.map sigmoid' zs)

  type LayerConfig =
    { weights: Matrix<float>; biases: Vector<float> }
    member this.length = Vector.length this.biases
    member this.batchBiases(batchLength) =
      DenseMatrix.ofColumnSeq(Enumerable.Repeat(this.biases, batchLength))
    static member rand(prevSize, size) =
      { weights = DenseMatrix.randomStandard<float> size prevSize / sqrt (float prevSize)
        biases = DenseVector.randomStandard<float> size }

  type DataPoint = { inputVector: Vector<float>; expectedVector: Vector<float> }

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

  type Network(layerSizes: int list) =
    let numLayers = layerSizes.Length

    let feedForwardLayer batchSize activations thisLayer =
      let zs = feedForward thisLayer.weights (List.head activations).activations (thisLayer.batchBiases(batchSize))
      Activations.withWeightedInputs(zs) :: activations

    let feedForwardBatch layers inputMatrix =
      let f = feedForwardLayer (Matrix.columnCount inputMatrix)
      List.fold f [Activations.inputLayer(inputMatrix)] layers

    let backPropLayer (nextLayer, thisActivations) errors =
      backPropErrors (nextLayer.weights.Transpose() * List.head errors) thisActivations.weightedInputs :: errors

    let calcWeightError (prevActivations, thisErrors) = thisErrors * prevActivations.activations.Transpose()

    let updateLayer factor (oldLayer, biasErrors, weightErrors) =
      { biases = oldLayer.biases - factor * Matrix.sumRows biasErrors
        weights = oldLayer.weights - factor * weightErrors }

    member this.randomLayers =
      let skipLast = Seq.take (numLayers - 1) layerSizes
      let skipFirst = Seq.skip 1 layerSizes
      Seq.zip skipLast skipFirst |> Seq.map LayerConfig.rand |> Seq.toList

    member this.evaluate(testData : Batch, layers) = List.head (feedForwardBatch layers testData.inputMatrix)

    member this.gradientDescent(train, test, epochs, batchSize, learningRate, initialLayers, evaluateFn) =
      let runBatch makeLayer layers (batch : Batch) =
        let activations = feedForwardBatch layers batch.inputMatrix
        let outputActivations = List.head activations
        let layerErrors = [backPropErrors (outputActivations.activations - batch.expectedMatrix) outputActivations.weightedInputs]

        let prevActivations = activations |> List.tail |> List.rev
        let currActivations = List.tail prevActivations
        let nextLayers = List.tail layers

        let biasErrors = Seq.foldBack backPropLayer (Seq.zip nextLayers currActivations) layerErrors
        let weightErrors = Seq.map calcWeightError (Seq.zip prevActivations biasErrors)

        Seq.zip3 layers biasErrors weightErrors |> Seq.map makeLayer |> Seq.toList

      let runEpoch initialLayers i =
        // Start stopwatch
        printf "Epoch %i" i
        let sw = System.Diagnostics.Stopwatch()
        sw.Start()

        // Get batches
        let batches =
          train
          |> Shuffle.shuffleList
          |> Seq.chunkBySize batchSize
          |> Seq.map (fun x -> { data = Seq.toList x })
          |> Seq.toList

        // Train the NN
        let newLayers = Seq.fold (runBatch (updateLayer (learningRate / float batchSize))) initialLayers batches |> Seq.toList
        printf " - Took %A" sw.Elapsed

        // Evaluate the new layers
        let testBatch = { data = test }
        evaluateFn testBatch.expectedMatrix (this.evaluate(testBatch, newLayers))

        // Return the new layers
        newLayers

      Seq.fold runEpoch initialLayers [ 1 .. epochs ]