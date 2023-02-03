using System.Collections;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using System.Threading.Tasks;
using Object = UnityEngine.Object;

[Serializable]
public struct Resolution
{
    public int width;
    public int height;
}

[Serializable]
public struct Range
{
    public float min;
    public float max;
}

public static class Utils
{
    public const int RHO_SALT_WATER = 1025;
    public const float metric_conversion_constant = 0.000016387f;      // 1 inch ^ 3 = 1.6387 * 10 ^ -5

    public static List<DirectoryInfo> Split(this DirectoryInfo path)
    {
        if(path == null)
        {
            throw new ArgumentNullException("path");
        } 

        List<DirectoryInfo> ret = new List<DirectoryInfo>();
        if (path.Parent != null)
        {
            ret.AddRange(Split(path.Parent));
        }
        
        ret.Add(path);
        return ret;
    }

    public static T[] SubArray<T>(this T[] array, int offset, int length)
    {
        Debug.Log(array);
        T[] result = new T[length];
        Array.Copy(array, offset, result, 0, length);
        return result;
    }

    public static float[] Vector3ToFloatArray(Vector3 vec)
    {
        return new float[] { vec.x, vec.y, vec.z };
    }

    public static Vector3 FloatArrayToVector3(float[] arr)
    {
        if(arr.Length != 3)
        {
            return Vector3.zero;
        }

        return new Vector3(arr[0], arr[1], arr[2]);
    }

    public static Quaternion FloatArrayToQuaternion(float[] arr)
    {
        if (arr.Length != 4)
        {
            return Quaternion.identity;
        }

        return new Quaternion(arr[0], arr[1], arr[2], arr[3]);
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

    public static void CleanAndCreateDirectories(Dictionary<string, bool> dir_paths)
    {
        foreach (KeyValuePair<string, bool> path in dir_paths)
        {
            if (path.Key == null)
            {
                continue;
            }

            if (Directory.Exists(path.Key))
            {
                if (path.Value)
                {
                    Directory.Delete(path.Key, true);
                    Directory.CreateDirectory(path.Key);
                }
            }

            else
            {
                Directory.CreateDirectory(path.Key);
            }
        }
    }

    public static void QuitApplication()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
        return;
#endif

        Application.Quit();
    }

    public static IEnumerator TakeScreenshot(Tuple<int, int> res, Camera cam, DirectoryInfo dir, System.Action<byte[]> callback = null)
    {
        RenderTexture prevRenderTexture = RenderTexture.active;
        RenderTexture prevCameraTargetTexture = cam.targetTexture;
        bool prevCameraEnabled = cam.enabled;
        cam.enabled = false;

        yield return new WaitForEndOfFrame();

        RenderTexture rt = RenderTexture.GetTemporary(res.Item1, res.Item2, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Default, 1);
        cam.targetTexture = rt;
        cam.Render();
        RenderTexture.active = rt;
        Texture2D screen_shot = new Texture2D(res.Item1, res.Item2, TextureFormat.RGB24, false);
        screen_shot.ReadPixels(new Rect(0, 0, res.Item1, res.Item2), 0, 0);
        screen_shot.Apply();
        cam.targetTexture = null;
        RenderTexture.active = null;
        Object.Destroy(screen_shot);

        byte[] byte_image = screen_shot.EncodeToJPG(100);

        if (dir != null)
        {
            File.WriteAllBytes(dir.FullName, byte_image);
        }

        if (callback != null)
        {
            callback.Invoke(byte_image);
        }

        cam.enabled = true;

        RenderTexture.ReleaseTemporary(rt);

        // restore modified data
        RenderTexture.active = prevRenderTexture;
        cam.targetTexture = prevCameraTargetTexture;
        cam.enabled = prevCameraEnabled;
        yield return null;
    }
}
