namespace FSharp.NeuralNet

open MathNet.Numerics.LinearAlgebra
open Network

module LoadData =

  let parseLine (line:string) =
    let makeExpectedVector result =
      DenseVector.init 10 (fun x -> if x = result then 1.0 else 0.0)

    let split = line.Split(':')
    let expectedResult = int split.[1]
    let inputVector =
      split.[0].Split(',')
      |> Seq.map float
      |> DenseVector.ofSeq
    let expectedVector = makeExpectedVector expectedResult
    { inputVector = inputVector; expectedVector = expectedVector }

  let load_data path =
    System.IO.File.ReadLines(path) |> Seq.map parseLine
