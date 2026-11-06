// Avatar Studio as one file: this small program carries the whole app (a zip appended to the end of this exe).
// First start: unpack it once into %LOCALAPPDATA%\AvatarStudio\app\<build> (a progress window shows). Every start:
// run the app from there with the same arguments and return its exit code. Unpacking once instead of on every
// start (PyInstaller's one-file mode unpacks ~800 MB to a temp folder each time) keeps later starts instant.
// Built by build_dist.py with the C# compiler that comes with Windows (.NET Framework 4.x).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Windows.Forms;

static class SingleExe
{
    const string AppExe = "AvatarStudio.exe";
    const int KeepBuilds = 2;  // unpacked builds kept in the cache: this one and the previous (older ones removed)

    [STAThread]
    static int Main(string[] args)
    {
        string self = Assembly.GetExecutingAssembly().Location;
        string root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                                   "AvatarStudio", "app");
        string dir;
        try
        {
            // the version is the zip comment at the very end of this file: reading 128 bytes is much quicker
            // than opening the whole zip directory (~2900 entries) on every start
            string build = TailBuild(self);
            if (build == null)
            {
                using (var zip = ZipFile.OpenRead(self))
                {
                    var entry = zip.GetEntry("BUILD");
                    if (entry == null) throw new InvalidDataException("this file doesn't contain the app");
                    using (var r = new StreamReader(entry.Open())) build = r.ReadToEnd().Trim();
                }
            }
            dir = Path.Combine(root, build);
            if (!File.Exists(Path.Combine(dir, ".ready")))
            {
                // started twice at once on a first start: the second waits for the first one's unpacking
                using (var mutex = new Mutex(false, "AvatarStudio-unpack-" + build))
                {
                    try { mutex.WaitOne(); } catch (AbandonedMutexException) { }
                    try
                    {
                        if (!File.Exists(Path.Combine(dir, ".ready")) && !Unpack(self, root, build))
                            return 1;
                    }
                    finally { mutex.ReleaseMutex(); }
                }
            }
        }
        catch (Exception e)
        {
            MessageBox.Show("Avatar Studio could not start:\n\n" + e.Message, "Avatar Studio",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }

        var psi = new ProcessStartInfo(Path.Combine(dir, AppExe), Quote(args))
        {
            WorkingDirectory = dir,
            UseShellExecute = false,
            // the launcher, viewer and settings windows need no console; other tools get one for their messages
            CreateNoWindow = args.Length == 0 || args[0] == "viewer" || args[0] == "settings",
        };
        psi.EnvironmentVariables["AVATARSTUDIO_EXE"] = self;  // where this file is (WiLoR helper lookup)
        try
        {
            using (var p = Process.Start(psi))
            {
                p.WaitForExit();
                return p.ExitCode;
            }
        }
        catch (Exception e)
        {
            MessageBox.Show("Avatar Studio could not start:\n\n" + e.Message, "Avatar Studio",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }

    static string TailBuild(string self)
    {
        try
        {
            using (var f = new FileStream(self, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            {
                int n = (int)Math.Min(128, f.Length);
                f.Seek(-n, SeekOrigin.End);
                var buf = new byte[n];
                f.Read(buf, 0, n);
                string tail = Encoding.ASCII.GetString(buf);
                int i = tail.LastIndexOf("AVSBUILD=");
                if (i < 0) return null;
                string b = tail.Substring(i + 9).Trim('\0', ' ', '\r', '\n');
                return b.Length > 0 && b.IndexOfAny(Path.GetInvalidFileNameChars()) < 0 ? b : null;
            }
        }
        catch { return null; }
    }

    // Unpack into <root>\<build>.partial, then rename it into place, so a cancelled or failed first start never
    // leaves a half-unpacked app behind that looks complete.
    static bool Unpack(string self, string root, string build)
    {
        Directory.CreateDirectory(root);
        string final = Path.Combine(root, build), part = final + ".partial";
        var form = new Form
        {
            Text = "Avatar Studio", Width = 460, Height = 150, FormBorderStyle = FormBorderStyle.FixedDialog,
            MaximizeBox = false, MinimizeBox = false, StartPosition = FormStartPosition.CenterScreen,
        };
        var label = new Label { Text = "Preparing Avatar Studio (first start of this version only)...",
                                Left = 16, Top = 16, Width = 420 };
        var bar = new ProgressBar { Left = 16, Top = 48, Width = 410, Height = 22, Maximum = 1000 };
        form.Controls.Add(label);
        form.Controls.Add(bar);
        Exception error = null;
        var worker = new Thread(() =>
        {
            try
            {
                if (Directory.Exists(part)) Directory.Delete(part, true);
                using (var zip = ZipFile.OpenRead(self))
                {
                    var files = zip.Entries.Where(e => e.FullName.StartsWith("app/") && !e.FullName.EndsWith("/")).ToList();
                    long total = Math.Max(1, files.Sum(e => e.Length)), done = 0;
                    foreach (var e in files)
                    {
                        string target = Path.Combine(part, e.FullName.Substring(4).Replace('/', '\\'));
                        Directory.CreateDirectory(Path.GetDirectoryName(target));
                        e.ExtractToFile(target, true);
                        done += e.Length;
                        int v = (int)(done * 1000 / total);
                        form.BeginInvoke((Action)(() => bar.Value = Math.Min(v, 1000)));
                    }
                }
                File.WriteAllText(Path.Combine(part, ".ready"), build);
                if (Directory.Exists(final)) Directory.Delete(final, true);
                Directory.Move(part, final);
                Prune(root, build);
            }
            catch (Exception ex) { error = ex; }
            form.BeginInvoke((Action)(() => form.Close()));
        });
        form.Shown += (s, e) => worker.Start();
        Application.EnableVisualStyles();
        Application.Run(form);
        worker.Join();
        if (error != null)
        {
            try { if (Directory.Exists(part)) Directory.Delete(part, true); } catch { }
            MessageBox.Show("Avatar Studio could not be prepared:\n\n" + error.Message +
                            "\n\n(Is the disk full? It needs about 1 GB in " + root + ")", "Avatar Studio",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
            return false;
        }
        return true;
    }

    // Keep the newest KeepBuilds unpacked builds; one that is still running can't be deleted and is kept.
    static void Prune(string root, string current)
    {
        var dirs = new DirectoryInfo(root).GetDirectories()
            .Where(d => d.Name != current && !d.Name.EndsWith(".partial"))
            .OrderByDescending(d => d.CreationTimeUtc).Skip(KeepBuilds - 1);
        foreach (var d in dirs)
        {
            try { d.Delete(true); } catch { }
        }
    }

    // Command-line quoting the way Windows programs (and Python) parse it back.
    static string Quote(string[] args)
    {
        var sb = new StringBuilder();
        foreach (var a in args)
        {
            if (sb.Length > 0) sb.Append(' ');
            if (a.Length > 0 && a.IndexOfAny(new[] { ' ', '\t', '"' }) < 0) { sb.Append(a); continue; }
            sb.Append('"');
            int slashes = 0;
            foreach (char c in a)
            {
                if (c == '\\') { slashes++; continue; }
                if (c == '"') { sb.Append('\\', slashes * 2 + 1); sb.Append('"'); }
                else { sb.Append('\\', slashes); sb.Append(c); }
                slashes = 0;
            }
            sb.Append('\\', slashes * 2);
            sb.Append('"');
        }
        return sb.ToString();
    }
}
