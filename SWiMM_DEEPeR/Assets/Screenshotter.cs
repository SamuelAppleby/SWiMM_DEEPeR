using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class Screenshotter : MonoBehaviour
{
    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct ScreenshotObject
    {
        public string name;
        public Vector3 position;
        public Vector3 rotation;
    }

    public List<ScreenshotObject> objects;

    public Transform screeshot_location;
    public Camera cam;

    private GameObject current_obj = null;

    DirectoryInfo dir;

    private int current_idx = -1;

    private Vector3 screenshot_loc = new Vector3(0, -0.3f, 1.25f);
    private Vector3 screenshot_rot = new Vector3(0, 60, 0);

    private void Start()
    {
        objects = new List<ScreenshotObject>()
        {
            new ScreenshotObject
            {
                name = "dolphin",
                position = screenshot_loc,
                rotation = new Vector3(0,180,0),
            },
            new ScreenshotObject
            {
                name = "rov",
                position = screenshot_loc + new Vector3(0, 0.25f, 0),
                rotation = new Vector3(-90,170,0),
            },
        };

        DirectoryInfo di = new DirectoryInfo(Path.GetFullPath(Directory.GetCurrentDirectory()));

#if UNITY_EDITOR
        dir = new DirectoryInfo(Path.GetFullPath(Path.Combine(di.FullName, "Screenshots")));
#else
        dir = new DirectoryInfo(Path.GetFullPath(Path.Combine(di.Parent.Parent.FullName, "Screenshots")));
#endif     

        Utils.CleanAndCreateDirectories(new Dictionary<DirectoryInfo, bool>()
            {
                { dir, false }
            });
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            if(current_obj != null)
            {
                Destroy(current_obj);
            }

            current_idx = current_idx == objects.Count - 1 ? 0 : current_idx+1;

            current_obj = Instantiate((GameObject)Resources.Load(objects[current_idx].name), cam.transform.position + objects[current_idx].position, Quaternion.Euler(objects[current_idx].rotation + screenshot_rot));
            current_obj.transform.parent = screeshot_location.transform;
        }

        if (Input.GetKeyDown(GlobalControlMap.Key_Screenshot))
        {
            StartCoroutine(Utils.TakeScreenshot(new Resolution { width = 7680, height = 4320 }, cam, new DirectoryInfo(Path.GetFullPath(Path.Combine(dir.FullName, current_obj.name + ".jpg")))));
        }

        if (Input.GetKeyDown(GlobalControlMap.Key_Quit))
        {
            Utils.QuitApplication();
        }
    }
}
