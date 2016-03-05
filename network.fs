namespace FSharp.NeuralNet

open MathNet.Numerics.LinearAlgebra

module Network =

  let sigmoid z = 1.0 / (1.0 + exp -z)

  type Layer = {
    weights: Matrix<float>
    biases: Matrix<float>
  }

  type Batch = {
    inputs: Matrix<float>
    expected_outputs: Matrix<float>
  }

  type DataPoint = {
    input_vector: Vector<float>
    expected_vector: Vector<float>
  }

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

  //  let gradient_descent training_data test_data num_epochs batch_size learning_rate =
  //    let run_batch batch_initial_layers batch =
  //      // Do gradient descent, back propogration and...
  //      // Return new layers
  //
  //    let run_epoch epoch_initial_layers _ =
  //      let batches : Batch list =
  //        // TODO: We need to make training_data a Tuple of (inputs, expected_outputs)
  //        // Where inputs and expected_outputs are matrices with each column a given data point.
  //        // So in this case inputs would be a 748x50,000 matrix (748 input nodes, 50,000 data points)
  //        //                 and expected_outputs would be a 10x50,000 matrix (10 output nodes, 50,000 data points)
  //        // Then we need to:
  //        //   1. Create a random Permutation to apply
  //        //   2. Apply it to both matrices to shuffle them
  //        //   3. Partition both matrices column wise by the batch_size, to get N
  //        //      Batch records (N=number of batches).
  //
  //      let trained_layers = Seq.fold run_batch epoch_initial_layers batches
  //      // TODO: evaluate trained_layers test_data
  //      trained_layers
  //
  //    let trained_layers =
  //      Seq.fold run_epoch random_layers seq { 1 .. num_epochs }
