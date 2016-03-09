namespace FSharp.NeuralNet

open MathNet.Numerics.LinearAlgebra
open System.Linq

module Network =

  let sigmoid z = 1.0 / (1.0 + exp -z)
  let sigmoid' z = sigmoid(z) * (1.0 - sigmoid(z))

  type LayerConfig =
    { weights: Matrix<float>; biases: Vector<float> }
    member this.length = Vector.length this.biases
    member this.batchBiases(batchLength) =
      DenseMatrix.ofColumnSeq(Enumerable.Repeat(this.biases, batchLength))
    static member rand(prevSize, size) =
      { weights = DenseMatrix.randomStandard<float> size prevSize
        biases = DenseVector.randomStandard<float> size }

  type DataPoint = { inputVector: Vector<float>; expectedVector: Vector<float> }

  type Batch =
    { data: DataPoint list }
    member private this.matrixOf f = this.data |> Seq.map f |> DenseMatrix.ofColumnSeq
    member this.inputMatrix = this.matrixOf (fun x -> x.inputVector)
    member this.expectedMatrix = this.matrixOf (fun x -> x.expectedVector)

  type Activations =
    { weightedInputs: Matrix<float>; activations: Matrix<float> }
    static member inputLayer(a) =
      { weightedInputs = a; activations = a }
    static member withWeightedInputs(z) =
      { weightedInputs = z; activations = Matrix.map sigmoid z }

  type Network(layerSizes: int list) =
    let numLayers = layerSizes.Length

    let randomLayers =
      let skipLast = Seq.take (numLayers - 1) layerSizes
      let skipFirst = Seq.skip 1 layerSizes
      Seq.zip skipLast skipFirst |> Seq.map LayerConfig.rand |> Seq.toList

    let feedForwardLayer batchSize (activations: Activations list) (thisLayer: LayerConfig)  =
        let zs = thisLayer.weights * (List.head activations).activations + thisLayer.batchBiases(batchSize)
        Activations.withWeightedInputs(zs) :: activations

    let feedForwardBatch (layers: LayerConfig list) inputMatrix =
      let f = feedForwardLayer (Matrix.columnCount inputMatrix)
      List.fold f [Activations.inputLayer(inputMatrix)] layers

    let makeErrors delta zs = 
        delta.* (Matrix.map sigmoid' zs)

    let updateLayer factor (oldLayer : LayerConfig, biasErrors : Matrix<float>, weightErrors: Matrix<float>) =
      { biases = oldLayer.biases - biasErrors.RowSums().Multiply(factor)
        weights = oldLayer.weights - weightErrors.Multiply(factor) }

    let backPropLayer (errors: Matrix<float> list) (nextLayer: LayerConfig, thisActivations: Activations) =
      makeErrors (nextLayer.weights.Transpose() * (List.head errors)) thisActivations.weightedInputs :: errors

    let calcWeightErrors (prevActivations, thisErrors) = thisErrors * prevActivations.activations.Transpose()

    let runBatch makeLayer layers (batch: Batch) =
        let activations = feedForwardBatch layers batch.inputMatrix
        let outputActivations = List.head activations
        let layerErrors = [makeErrors (outputActivations.activations - batch.expectedMatrix) outputActivations.weightedInputs]

        let allButLastActivations = activations |> List.tail |> List.rev
        let middleActivations = allButLastActivations |> List.tail
        let nextLayers = layers |> List.tail

        let biasErrors = Seq.fold backPropLayer layerErrors (Seq.zip (List.rev nextLayers) (List.rev middleActivations))
        let weightErrors = Seq.map calcWeightErrors (Seq.zip allButLastActivations biasErrors)

        Seq.zip3 layers biasErrors weightErrors |> Seq.map makeLayer |> Seq.toList

    member this.evaluate(inputMatrix) =
      feedForwardBatch randomLayers inputMatrix

    member this.gradientDescent(train, test, epochs, batchSize, learningRate) =

      let runEpoch initialLayers i =
        printfn "Epoch %i" i

        let batches =
          train
          |> Shuffle.shuffleList
          |> Seq.chunkBySize batchSize
          |> Seq.map (fun x -> { data = Seq.toList x })
          |> Seq.toList

        Seq.fold (runBatch (updateLayer (learningRate/float batchSize))) initialLayers batches
        // TODO: Evaluate after each epoch

      Seq.fold runEpoch randomLayers [ 1 .. epochs ]