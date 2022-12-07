using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(BoxCollider))]
public class NastyWater : MonoBehaviour {

    public int width, length, height;

    private Vector3[] vertices;


    private Mesh mesh;

    public void GenerateWater()
    {
        GetComponent<MeshFilter>().mesh = mesh = new Mesh();
        mesh.name = "Water";

        vertices = new Vector3[(width + 1) * (length + 1)];
        Vector2[] uv = new Vector2[vertices.Length];
        for (int i = 0, y = 0; y <= length; y++)
        {
            for (int x = 0; x <= width; x++, i++)
            {
                vertices[i] = new Vector3(x - width / 2.0f, 0, y - length / 2.0f);
                uv[i] = new Vector2((float)x / width, (float)y / length);
            }
        }
        mesh.vertices = vertices;
        mesh.uv = uv;

        int[] triangles = new int[width * length * 6];
        for (int ti = 0, vi = 0, y = 0; y < length; y++, vi++)
        {
            for (int x = 0; x < width; x++, ti += 6, vi++)
            {
                triangles[ti] = vi;
                triangles[ti + 3] = triangles[ti + 2] = vi + 1;
                triangles[ti + 4] = triangles[ti + 1] = vi + width + 1;
                triangles[ti + 5] = vi + width + 2;
            }
        }
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        mesh.RecalculateTangents();

        GetComponent<BoxCollider>().size = new Vector3(width,height,length);
        GetComponent<BoxCollider>().center = new Vector3(0,-height / 2.0f,0);
        GetComponent<BoxCollider>().isTrigger = true;

        GetComponent<MeshRenderer>().shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;

        print(GetComponent<BoxCollider>().size);
    }

}
