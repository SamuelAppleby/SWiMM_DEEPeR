using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class Utils
{
    public const int RHO_SALT_WATER = 1025;
    public const float metric_conversion_constant = 0.000016387f;      // 1 inch ^ 3 = 1.6387 * 10 ^ -5

    public static float[] Vector3ToFloatArray(ref Vector3 vec)
    {
        return new float[] { vec.x, vec.y, vec.z };
    }

    public static Vector3 FloatArrayToVector3(ref float[] arr)
    {
        return new Vector3(arr[0], arr[1], arr[2]);
    }

    public static float VolumeOfBoxCollider(Collider collider)
    {
        BoxCollider b_col = collider as BoxCollider;

        if(b_col != null)
        {
            return b_col.size.x * b_col.size.y * b_col.size.z;
        }

        return 0;
    }

    public static float SignedVolumeOfTriangle(Vector3 p1, Vector3 p2, Vector3 p3)
    {
        float v321 = p3.x * p2.y * p1.z;
        float v231 = p2.x * p3.y * p1.z;
        float v312 = p3.x * p1.y * p2.z;
        float v132 = p1.x * p3.y * p2.z;
        float v213 = p2.x * p1.y * p3.z;
        float v123 = p1.x * p2.y * p3.z;
        return (1.0f / 6.0f) * (-v321 + v231 + v312 - v132 - v213 + v123);
    }

    public static float VolumeOfMesh(Mesh mesh)
    {
        float volume = 0;
        Vector3[] vertices = mesh.vertices;
        int[] triangles = mesh.triangles;
        for (int i = 0; i < mesh.triangles.Length; i += 3)
        {
            Vector3 p1 = vertices[triangles[i + 0]];
            Vector3 p2 = vertices[triangles[i + 1]];
            Vector3 p3 = vertices[triangles[i + 2]];
            volume += SignedVolumeOfTriangle(p1, p2, p3);
        }
        return Mathf.Abs(volume);
    }
}
