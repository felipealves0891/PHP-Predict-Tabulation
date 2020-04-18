<?php
declare(strict_types=1);

namespace App;

class Neuron 
{
    /**
     * @var float
     */
    protected float $learningRate;

    /**
     * @var array
     */
    protected array $inputs;

    /**
     * Get the value of inputs
     *
     * @return  array
     */ 
    public function getInputs()
    {
        return $this->inputs;
    }

    /**
     * Set the value of inputs
     *
     * @param  array  $inputs
     *
     * @return  self
     */ 
    public function setInputs(array $inputs)
    {
        $this->inputs = $inputs;

        return $this;
    }
    
    /**
     * @var float
     */
    protected float $bias;
    
    /**
     * The bias weight is found in the first position (Weights[0])
     * @var float
     */
    protected array $weights = array();

    /**
     * Get the bias weight is found in the first position (Weights[0])
     *
     * @return  float
     */ 
    public function getWeights()
    {
        return $this->weights;
    }

    /**
     * Set the bias weight is found in the first position (Weights[0])
     *
     * @param  float  $weights  The bias weight is found in the first position (Weights[0])
     *
     * @return  self
     */ 
    public function setWeights(float $weights)
    {
        $this->weights = $weights;

        return $this;
    }
    
    /**
     * Output from the Neuron
     * @var float
     */
    protected float $outputs;

    /**
     * Get output from the Neuron
     *
     * @return  float
     */ 
    public function getOutputs()
    {
        return $this->outputs;
    }

    /**
     * Set output from the Neuron
     *
     * @param  float  $outputs  Output from the Neuron
     *
     * @return  self
     */ 
    public function setOutputs(float $outputs)
    {
        $this->outputs = $outputs;

        return $this;
    }
    
    /**
     * Instâncias anteriores com suas propriedades
     * @var float 
     */
    protected float $expectedOutput = 0.0;

    /**
     * Get instâncias anteriores com suas propriedades
     *
     * @return  float
     */ 
    public function getExpectedOutput()
    {
        return $this->expectedOutput;
    }

    /**
     * Set instâncias anteriores com suas propriedades
     *
     * @param  float  $expectedOutput  Instâncias anteriores com suas propriedades
     *
     * @return  self
     */ 
    public function setExpectedOutput(float $expectedOutput)
    {
        $this->expectedOutput = $expectedOutput;

        return $this;
    }
    
    /**
     * Instâncias anteriores com suas propriedades
     * @var float 
     */
    protected float $error = 0.0;

    /**
     * Get instâncias anteriores com suas propriedades
     *
     * @return  float
     */ 
    public function getError()
    {
        return $this->error;
    }

    /**
     * Set instâncias anteriores com suas propriedades
     *
     * @param  float  $error  Instâncias anteriores com suas propriedades
     *
     * @return  self
     */ 
    public function setError(float $error)
    {
        $this->error = $error;

        return $this;
    }
    
    /**
     * Instâncias anteriores com suas propriedades
     * @var float 
     */
    protected float $backPropagatedError = 0.0;

    /**
     * Get instâncias anteriores com suas propriedades
     *
     * @return  float
     */ 
    public function getBackPropagatedError()
    {
        return $this->backPropagatedError;
    }

    /**
     * Set instâncias anteriores com suas propriedades
     *
     * @param  float  $backPropagatedError  Instâncias anteriores com suas propriedades
     *
     * @return  self
     */ 
    public function setBackPropagatedError(float $backPropagatedError)
    {
        $this->backPropagatedError = $backPropagatedError;

        return $this;
    }
    
    /**
     * @param float $learningRate
     * @param array $inputs
     */
    public function __construct(float $learningRate, array $inputs, float $bias = 1.0)
    {
        $this->learningRate = $learningRate;
        $this->inputs = $inputs;
        $this->bias = $bias;
        $this->sortWeights();
    }

    /**
     * Sorteia o peso para cada entrada incluindo o bias
     */
    public function sortWeights() : void
    {
        for($i=0; $i <= count($this->inputs); $i++)
            if(isset($this->weights[$i]) == false)
                $this->weights[$i] = rand(-1000, 1000) / 1000;
    }

    public function calculateError() : void
    {
        $this->error = $this->expectedOutput - $this->outputs;
    }

    public function calculateBackPropagatedError() : void
    {
        $this->backPropagatedError = (1.0 - $this->outputs * $this->outputs) * $this->error;
    }

    public function weightsAdjustment() : void
    {
        $this->weights[0] += $this->learningRate * $this->bias * $this->backPropagatedError;
        for ($i = 1; $i < count($this->weights); $i++)
            $this->weights[$i] += $this->learningRate * $this->inputs[$i - 1] * $this->backPropagatedError;
    }

    /**
     * Realiza o processo de propagação
     */
    public function forward() : void
    {
        $sum = $this->bias * $this->weights[0]; 

        for($i=0; $i < count($this->inputs); $i++)
            $sum += $this->inputs[$i] * $this->weights[$i+1];

        $this->outputs = tan($sum);
    }

    private function sigmoid(float $t) : float
    {
        return 1 / (1 - exp(-$t));
    }

    /**
     * Realiza o processo de retro-propagação
     */
    public function backward(array $inputs, float $expectedOutput) : void
    {
        $this->inputs = $inputs;
        $this->expectedOutput = $expectedOutput;

        $this->forward();

        $this->calculateError();
        $this->calculateBackPropagatedError();

        $this->weightsAdjustment();
    }

}