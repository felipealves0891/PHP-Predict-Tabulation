<?php
declare(strict_types=1);

namespace App;

class PerceptronNetwork 
{
    /**
     * @var array
     */
    protected array $inputs;
    
    /**
     * @var array
     */
    protected array $neuron = array();

    /**
     * @param array $inputs
     * @param int $neuronsAmount = 3
     */
    public function __construct(array $inputs, int $neuronsAmount = 3)
    {
        $this->inputs = $inputs;
        for($i=0; $i <= $neuronsAmount; $i++) 
            $this->neurons[] = new Neuron(0.1, $inputs);
    }

    public function getOutputs() : array
    {
        $outputs = array();
        for($i=0; $i < count($this->neurons); $i++)
            $outputs[$i] = $this->neurons[$i];
        
        return $outputs;
    }

    public function forward() 
    {
        foreach ($this->neurons as $key => $neuron) 
            $neuron->forward();
    }
}