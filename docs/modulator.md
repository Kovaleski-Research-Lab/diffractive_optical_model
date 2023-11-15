## ModulatorFactory

The modulator factory will handle selecting the correct modulator from the 
params. Just want to abstract the logic from the implementations.

Requirements:
plane - a plane from plane.py
params - dict
    type (str) - one of types below
    phase_init (str) - one of the phase initializations below
    amplitude_init (str) - one of the amplitude initializations below
    phase_pattern (optional: str) - used with defined initialization
    amplitude_pattern (optional: str) - used with defined initialization

## Types of modulators
Modulator type sets the gradients for back propagation. This is so that we 
can selectively optimize the amplitude, the phase, or both amplitude and 
phase of the modulator.

1. phase_only
  1. Only the phase is optimizeable
2. amplitude_only
  1. Only the amplitude is optimizeabel
3. complex
  1. Both amplitude and phase are optimizeable
4. None
  1. No optimization

## Initializations
The initializations set what the modulator values are on creation. For optimization
tasks this sets the starting point. These values are passed as either
phase_init or amp_init in the params.

1. random
  1. Amplitude range[0, 1], torch.rand()
  2. Phase range[0, 2pi], torch.rand()
2. uniform
  1. Amplitude [1]
  2. Phase [0]
3. lens
  1. Amplitude [1]
  2. Phase range[0, 2pi], LensPhaseFunction()
4. custom
  1. Amplitude range[0, 1], customFunction()
  2. Phase range[0, 2pi], customFunction()


