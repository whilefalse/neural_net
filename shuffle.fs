namespace FSharp.NeuralNet

open MathNet.Numerics.LinearAlgebra

module Shuffle =
  let rand = new System.Random()

  let swap (a: _[]) x y =
    let tmp = a.[x]
    a.[x] <- a.[y]
    a.[y] <- tmp

  // shuffle an array (in-place)
  let shuffleArray a =
    Array.iteri (fun i _ -> swap a i (rand.Next(i, Array.length a))) a
    a

  let shuffleList l =
    l |> List.toArray |> shuffleArray |> Array.toList
