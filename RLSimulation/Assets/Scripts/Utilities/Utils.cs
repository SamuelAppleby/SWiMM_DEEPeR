using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class Utils
{
    public static float[] Vector3ToFloatArray(ref Vector3 vec)
    {
        return new float[] { vec.x, vec.y, vec.z };
    }

    public static Vector3 FloatArrayToVector3(ref float[] arr)
    {
        return new Vector3(arr[0], arr[1], arr[2]);
    }
}
