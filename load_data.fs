namespace FSharp.NeuralNet

open MathNet.Numerics.LinearAlgebra
open Network

module LoadData =

  let parse_line (line:string) =
    let make_expected_vector result =
      SparseVector.init 10 (fun x -> if x = result then 1.0 else 0.0)

    let split = line.Split(':')
    let expected_result = int split.[1]
    let input_vector =
      split.[0].Split(',')
      |> Seq.map float
      |> SparseVector.ofSeq
    let expected_vector = make_expected_vector expected_result
    { input_vector = input_vector; expected_vector = expected_vector }

  let load_data path =
    System.IO.File.ReadLines(path) |> Seq.map parse_line
