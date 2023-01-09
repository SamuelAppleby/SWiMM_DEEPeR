using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(NastyWater))]
[CanEditMultipleObjects]
public class WaterEditor : Editor
{
    private GUISkin Skin;
    private int width = 0,length = 0, height = 0;

    private NastyWater _target;

    private void OnEnable()
    {
        _target = target as NastyWater;
        Skin = Resources.Load("NastySkin") as GUISkin;
    }

    // Update is called once per frame
    void Update()
    {
        GUI.skin = Skin;
    }

    public override void OnInspectorGUI()
    {
        GUI.skin = Skin;
        EditorGUILayout.BeginVertical("box");
            EditorGUILayout.BeginVertical("box");
            width = EditorGUILayout.IntField(new GUIContent("Width:"), width);
            length = EditorGUILayout.IntField(new GUIContent("Length:"),length);
            height = EditorGUILayout.IntField(new GUIContent("Height:"), height);
        EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginVertical("box");
        if (GUILayout.Button(new GUIContent("GENERATE MESH")))
        {
            _target.width =  width;
            _target.length = length;
            _target.height = height;
            _target.GenerateWater();
        }
            EditorGUILayout.EndHorizontal();
        EditorGUILayout.EndHorizontal();
    }

    private void OnSceneGUI()
    {

        Handles.color = Color.yellow;
        Handles.DrawWireCube(_target.transform.position + (Vector3.up * -height / 2.0f), new Vector3(width,height,length));
    }
}
