<?php
declare(strict_types=1);

namespace App;

use App\Neuron;

class MultiLayerPerceptronNetwork 
{
    /**
     * @var float
     */
    protected float $LearningRate;

    /**
     * @var array
     */
    protected array $inputs = array();

    /**
     * @var array
     */
    protected array $hiddenLayer = array();

    /**
     * @var array
     */
    protected array $outLayer = array();

    /**
     * @var array
     */
    protected array $expectedOutput = array();
    
    /**
     * @param array $inputs
     * @param int $hiddenLayerAmount
     * @param int $outLayerAmount
     * @param float $LearningRate
     */
    public function __construct(array $inputs, int $hiddenLayerAmount, int $outLayerAmount, float $LearningRate)
    {
        $this->LearningRate = $LearningRate;
        $this->inputs = $inputs;

        for($i=0; $i<$hiddenLayerAmount; $i++)
            $this->hiddenLayer[$i] = new Neuron($LearningRate, $inputs);

        for($i=0; $i<$outLayerAmount; $i++)
            $this->outLayer[$i] = new Neuron($LearningRate, $inputs);
    }

    public function getOutputs() : array
    {
        $output = array();
        for($j=0; $j < count($this->outLayer); $j++) 
            $output[] = $this->outLayer[$j]->getOutputs();

        return $output;
    }

    public function setInputOnHiddenLayer(array $inputs) 
    {
        for($j=0; $j < count($this->hiddenLayer); $j++) 
        {
            $this->hiddenLayer[$j]->setInputs($inputs);
            $this->hiddenLayer[$j]->sortWeights();
        }
            
    }

    public function setInputOnOutLayer(array $inputs) 
    {
        for($j=0; $j < count($this->outLayer); $j++) 
        {
            $this->outLayer[$j]->setInputs($inputs);
            $this->outLayer[$j]->sortWeights();
        }
            
    }

    private function CalculateHiddenLayerErrors() : void
    {
        for($i=0; $i < count($this->hiddenLayer); $i++) 
        {
            $sum = 0.0;
            for($j=0; $j < count($this->outLayer); $j++) 
            {
                $backPropagatedError = $this->outLayer[$j]->getBackPropagatedError();
                $weight = $this->outLayer[$j]->getWeights()[$i + 1];

                $sum += $backPropagatedError * $weight;
            }

            $this->hiddenLayer[$i]->setError($sum);

            $this->hiddenLayer[$i]->CalculateBackPropagatedError();
        }
    }

    /**
     * Realiza o processo de propagação (forward) na rede
     */
    public function forward() 
    {
        $tempOut = array();

        for($i=0; $i < count($this->hiddenLayer); $i++) 
        {
            $this->hiddenLayer[$i]->forward();
            $tempOut[$i] = $this->hiddenLayer[$i]->getOutputs();
        }

        $this->setInputOnOutLayer($tempOut);

        for($i=0; $i < count($this->outLayer); $i++)
            $this->outLayer[$i]->forward();

    }

    /**
     * Realiza o processo de retro-propagação (backward) na rede
     */
    public function Backward(array $inputs, array $expectedOutput) 
    {
        if(count($expectedOutput) == count($this->outLayer)) 
        {
            $this->inputs = $inputs;
            $this->setInputOnHiddenLayer($inputs);

            $this->expectedOutput = $expectedOutput;

            $this->forward();

            for ($i=0; $i < count($this->outLayer); $i++) 
            { 
                $this->outLayer[$i]->setExpectedOutput($expectedOutput[$i]);
                $this->outLayer[$i]->calculateError();
                $this->outLayer[$i]->calculateBackPropagatedError();
            }

            $this->CalculateHiddenLayerErrors();

            for($j=0; $j < count($this->hiddenLayer); $j++) 
                $this->hiddenLayer[$j]->WeightsAdjustment();

            for($j=0; $j < count($this->outLayer); $j++) 
                $this->outLayer[$j]->WeightsAdjustment();

        }
    }
}