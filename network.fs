namespace FSharp.NeuralNet

open MathNet.Numerics.LinearAlgebra

module Network =

  let sigmoid z = 1.0 / (1.0 + exp -z)

  type Layer = { weights: Matrix<float>; biases: Matrix<float> }
  type Batch = { inputMatrix: Matrix<float>; expectedMatrix: Matrix<float> }
  type DataPoint =
    { inputVector: Vector<float>; expectedVector: Vector<float> }
    member this.inputMatrix = this.inputVector.ToColumnMatrix()
    member this.expectedMatrix = this.expectedVector.ToColumnMatrix()

  type Network(layer_sizes: int list) =
    let num_layers = layer_sizes.Length

    let random_layers =
      let make_layer (last_layer_size, layer_size) = {
        weights = DenseMatrix.randomStandard<float> last_layer_size layer_size
        biases = DenseMatrix.randomStandard<float> layer_size 1
      }
      let skip_last = Seq.take (num_layers - 1) layer_sizes
      let skip_first = Seq.skip 1 layer_sizes
      Seq.zip skip_last skip_first |> Seq.map make_layer

    let feed_forward layers initial_inputs =
      let process_layer layer_inputs layer =
        (Matrix.transpose layer.weights) * layer_inputs + layer.biases
        |> Matrix.map sigmoid
      Seq.fold process_layer initial_inputs layers

    member this.evaluate(inputs) =
      feed_forward random_layers inputs

    member this.gradientDescent(training_data, test_data, numEpochs, batchSize, learning_rate) =
      let run_batch batch_initial_layers batch =
        batch_initial_layers
        // Do gradient descent, back propogration and...
        // Return new layers

      let run_epoch epoch_initial_layers i =
        printfn "Epoch %i" i
        let makeBatch dataPoints =
          let inputVectors = dataPoints |> Seq.map (fun x -> x.inputVector)
          let expectedVectors = dataPoints |> Seq.map (fun x -> x.expectedVector)
          {
            inputMatrix = SparseMatrix.ofColumnSeq(inputVectors)
            expectedMatrix = SparseMatrix.ofColumnSeq(expectedVectors)
          }

        let batches =
          // TODO: randomize training data before chunking it
          training_data
          |> Seq.chunkBySize batchSize
          |> Seq.map makeBatch

        // TODO: Evaluate after each epoch
        Seq.fold run_batch epoch_initial_layers batches

      let epochs = seq { 1 .. numEpochs }
      Seq.fold run_epoch random_layers epochs
