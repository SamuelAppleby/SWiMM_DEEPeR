using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using static Server;

/* Only require to add a new event type here if custom parameter is required */

[Serializable]
public class JsonDataUnityEvent: UnityEvent<JsonMessage>
{
}

[Serializable]
public class ROVInitialisedUnityEvent : UnityEvent<GameObject>
{
}

[Serializable]
public class ExceptionUnityEvent : UnityEvent<Exception>
{
}

[Serializable]
public class ServerConnectingUnityEvent : UnityEvent<string, int>
{
}

[Serializable]
public class SceneChangedUnityEvent : UnityEvent<Enums.E_SceneIndices>
{
}