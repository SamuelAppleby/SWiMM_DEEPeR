using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using static Server;

/* Only require to add a new event type here if custom parameter is required */

[Serializable]
public class JsonControlUnityEvent: UnityEvent<JsonMessage<JsonControls>>
{
}

[Serializable]
public class ROVInitialisedUnityEvent : UnityEvent<GameObject>
{
}
