using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu]
public class ObservationSentEvent : GameEvent
{
    public void Raise()
    {
        foreach (SimulationGameEventListener l in listeners)
        {
            l.OnObservationSentEventRaised();
        }
    }
}