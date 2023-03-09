using System.Collections;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Object = UnityEngine.Object;
using Unity.VisualScripting;

[Serializable]
public struct Resolution
{
    public int width;
    public int height;

    public override bool Equals(object? obj) => obj is Resolution other && this.Equals(other);

    public bool Equals(Resolution p) => width == p.width && height == p.height;

    public override int GetHashCode() => (width, height).GetHashCode();

    public static bool operator ==(Resolution lhs, Resolution rhs) => lhs.Equals(rhs);

    public static bool operator !=(Resolution lhs, Resolution rhs) => !(lhs == rhs);

    public override string ToString()
    {
        return width + "x" + height;
    }
}

[Serializable]
public struct Range
{
    public float min;
    public float max;
}

public static class Utils
{
    public static Resolution SAMPLE_RES = new Resolution()
    {
        width = 64,
        height = 64
    };

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

    public static void CleanAndCreateDirectories(Dictionary<DirectoryInfo, bool> dir_paths)
    {
        foreach (KeyValuePair<DirectoryInfo, bool> path in dir_paths)
        {
            if (path.Key == null)
            {
                continue;
            }

            if (path.Key.Exists)
            {
                if (path.Value)
                {
                    Directory.Delete(path.Key.FullName, true);
                    Directory.CreateDirectory(path.Key.FullName);
                }
            }

            else
            {
                Directory.CreateDirectory(path.Key.FullName);
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

    public static IEnumerator TakeScreenshot(Resolution res, Camera cam, DirectoryInfo dir, Action<byte[]> callback = null)
    {
        RenderTexture prev_render_tex = RenderTexture.active;
        RenderTexture prev_cam_target_tex = cam.targetTexture;
        bool prev_cam_enabled = cam.enabled;
        cam.enabled = false;

        yield return new WaitForEndOfFrame();

        RenderTexture rt = RenderTexture.GetTemporary(res.width, res.height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Default, 1);
        cam.targetTexture = rt;
        cam.Render();
        RenderTexture.active = rt;
        Texture2D screen_shot = new Texture2D(res.width, res.height, TextureFormat.RGB24, false);
        screen_shot.ReadPixels(new Rect(0, 0, res.width, res.height), 0, 0);
        screen_shot.Apply();

        byte[] byte_image = screen_shot.EncodeToJPG(100);

        if (dir != null)
        {
            File.WriteAllBytes(Path.GetFullPath(dir.FullName), byte_image);

            if(SimulationManager._instance.game_state == Enums.E_Game_State.VAE_GEN && res != SAMPLE_RES)
            {
                string sampled_dir_path = "";
                string[] subs = dir.FullName.Split(Path.DirectorySeparatorChar);

                for (int i = 0; i < subs.Length - 2; i++)
                {
                    sampled_dir_path += subs[i] + Path.DirectorySeparatorChar;
                }

                sampled_dir_path += SAMPLE_RES.ToString() + Path.DirectorySeparatorChar;

                DirectoryInfo sampled_dir = new DirectoryInfo(sampled_dir_path);

                if (!sampled_dir.Exists)
                {
                    Directory.CreateDirectory(sampled_dir.FullName);
                }

                int num = (Directory.GetFiles(sampled_dir.FullName).Length + 1);
                DirectoryInfo new_dir = new DirectoryInfo(sampled_dir.FullName + num.ToString() + ".jpg");

                RenderTexture rt_1 = RenderTexture.GetTemporary(SAMPLE_RES.width, SAMPLE_RES.height, 24);
                Graphics.Blit(screen_shot, rt_1);
                RenderTexture.active = rt_1;
                Texture2D result = new Texture2D(SAMPLE_RES.width, SAMPLE_RES.height);
                result.ReadPixels(new Rect(0, 0, SAMPLE_RES.width, SAMPLE_RES.height), 0, 0);
                result.Apply();

                File.WriteAllBytes(Path.GetFullPath(new_dir.FullName), result.EncodeToJPG(100));

                RenderTexture.ReleaseTemporary(rt_1);
                Object.Destroy(result);
            }
        }

        if (callback != null)
        {
            callback.Invoke(byte_image);
        }

        RenderTexture.ReleaseTemporary(rt);
        Object.Destroy(screen_shot);

        // restore modified data
        RenderTexture.active = prev_render_tex;
        cam.targetTexture = prev_cam_target_tex;
        cam.enabled = prev_cam_enabled;
        yield return null;
    }

    public static void Swap<T>(this List<T> list, int i, int j)
    {
        T temp = list[i];
        list[i] = list[j];
        list[j] = temp;
    }
}
