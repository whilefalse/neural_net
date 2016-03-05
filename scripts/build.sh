DLLS=(
  'lib/FSharp.Core.4.0.0.1/lib/net40/FSharp.Core.dll'
  'lib/MathNet.Numerics.3.11.0/lib/net40/MathNet.Numerics.dll'
  'lib/MathNet.Numerics.FSharp.3.11.0/lib/net40/MathNet.Numerics.FSharp.dll'
  'lib/TaskParallelLibrary.1.0.2856.0/lib/Net35/System.Threading.dll'
  )

CMD="fsharpc shuffle.fs network.fs load_data.fs neural_net.fs --out:bin/neural_net.exe --noframework --optimize --debug --standalone ${DLLS[@]/lib/-r:lib}"
echo $CMD
$CMD

for dll in ${DLLS[@]}
do
  CMD="cp $dll bin/"
  echo $CMD
  $CMD
done
