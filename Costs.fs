namespace FSharp.NeuralNet

open MathNet.Numerics.LinearAlgebra

module Costs = 
  let sigmoid z = 1.0 / (1.0 + exp -z)
  let sigmoid' z = sigmoid(z) * (1.0 - sigmoid(z))

  let crossEntropy (z: Matrix<float>) (a: Matrix<float>) (y: Matrix<float>) = a - y
  let quadratic (z: Matrix<float>) (a: Matrix<float>) (y: Matrix<float>) = (a-y).* Matrix.map sigmoid' z

