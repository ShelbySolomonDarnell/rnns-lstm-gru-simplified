clear

if [ ! -f logs/werernns.log ]; then
  echo 'No file to remove.'
else
  rm logs/werernns.log
fi

#CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python testrnns.py
#CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python trainrnns.py


runCode()
{
  read -p "Enter 0 to train and 1 to test --> " testing
  if [ $testing -eq 0 ]; then
    echo 'Let us train!'
    CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python trainrnns.py
  else
    echo 'Let us test!'
    CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python testrnns.py
  fi
}

runCode
