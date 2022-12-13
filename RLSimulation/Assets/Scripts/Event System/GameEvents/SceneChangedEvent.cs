using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu]
public class SceneChangedEvent : GameEvent
{
    public void Raise(Enums.E_SceneIndices to)
    {
        foreach (GameEventListener l in listeners)
        {
            l.OnSceneChangedEventRaised(to);
        }
    }
}
