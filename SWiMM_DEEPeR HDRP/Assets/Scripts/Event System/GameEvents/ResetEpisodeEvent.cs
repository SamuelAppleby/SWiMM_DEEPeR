using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu]
public class ResetEpisodeEvent : GameEvent
{
    public void Raise()
    {
        foreach (GameEventListener l in listeners)
        {
            l.OnEpisodeResetEventRaised();
        }
    }
}
