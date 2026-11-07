// Avatar Studio as one file: this small program carries the whole app (a zip appended to the end of this exe).
// First start: unpack it once into %LOCALAPPDATA%\AvatarStudio\app\<build> (a progress window shows, and stays until
// the app's own window is up, so nothing goes blank in between). Every start: run the app from there with the same
// arguments and return its exit code. Unpacking once instead of on every
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
        string root = Environment.GetEnvironmentVariable("AVATARSTUDIO_UNPACK_DIR");  // (tests: another cache)
        if (string.IsNullOrEmpty(root))
            root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                                "AvatarStudio", "app");
        string dir;
        Process started = null;  // (a first start starts the app from the progress window)
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
                        if (!File.Exists(Path.Combine(dir, ".ready")) &&
                            !Unpack(self, root, build, () => Start(dir, args, self), Windowed(args), out started))
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

        try
        {
            using (var p = started ?? Start(dir, args, self))
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

    // the app window, the viewer and settings windows (the other tools run in a console)
    static bool Windowed(string[] args)
    {
        return args.Length == 0 || args[0] == "viewer" || args[0] == "settings";
    }

    static Process Start(string dir, string[] args, string self)
    {
        var psi = new ProcessStartInfo(Path.Combine(dir, AppExe), Quote(args))
        {
            WorkingDirectory = dir,
            UseShellExecute = false,
            // the launcher, viewer and settings windows need no console; other tools get one for their messages
            CreateNoWindow = Windowed(args),
        };
        psi.EnvironmentVariables["AVATARSTUDIO_EXE"] = self;  // where this file is (WiLoR helper lookup)
        return Process.Start(psi);
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
    // leaves a half-unpacked app behind that looks complete. Several threads unpack at once (each with its own
    // reader of the zip): decompressing is most of the time, and one core did it all before (7.4 s -> 3.2 s here).
    // Then the app is started (start()) and the window stays until the app's window shows (or 60 s).
    static bool Unpack(string self, string root, string build, Func<Process> start, bool waitWindow,
                       out Process started)
    {
        started = null;
        Process app = null;
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
        var startError = (Exception)null;
        form.Controls.Add(label);
        form.Controls.Add(bar);
        Exception error = null;
        var worker = new Thread(() =>
        {
            try
            {
                if (Directory.Exists(part)) Directory.Delete(part, true);
                List<KeyValuePair<string, long>> files;
                using (var zip = ZipFile.OpenRead(self))
                    files = zip.Entries.Where(e => e.FullName.StartsWith("app/") && !e.FullName.EndsWith("/"))
                               .Select(e => new KeyValuePair<string, long>(e.FullName, e.Length)).ToList();
                long total = Math.Max(1, files.Sum(e => e.Value)), done = 0;
                int n = Math.Max(1, Math.Min(Environment.ProcessorCount, 6));
                var parts = Enumerable.Range(0, n).Select(_ => new List<string>()).ToArray();
                var load = new long[n];
                foreach (var f in files.OrderByDescending(f => f.Value))  // (biggest first, each to the least busy)
                {
                    int i = Array.IndexOf(load, load.Min());
                    parts[i].Add(f.Key);
                    load[i] += Math.Max(f.Value, 4096);
                }
                var errors = new List<Exception>();
                var threads = parts.Select(names => new Thread(() =>
                {
                    try
                    {
                        using (var zip = ZipFile.OpenRead(self))
                        {
                            foreach (string name in names)
                            {
                                var e = zip.GetEntry(name);
                                string target = Path.Combine(part, name.Substring(4).Replace('/', '\\'));
                                Directory.CreateDirectory(Path.GetDirectoryName(target));
                                e.ExtractToFile(target, true);
                                int v = (int)(Interlocked.Add(ref done, e.Length) * 1000 / total);
                                form.BeginInvoke((Action)(() => bar.Value = Math.Max(bar.Value, Math.Min(v, 1000))));
                            }
                        }
                    }
                    catch (Exception ex) { lock (errors) errors.Add(ex); }
                })).ToList();
                threads.ForEach(t => t.Start());
                threads.ForEach(t => t.Join());
                if (errors.Count > 0) throw errors[0];
                File.WriteAllText(Path.Combine(part, ".ready"), build);
                if (Directory.Exists(final)) Directory.Delete(final, true);
                Directory.Move(part, final);
                Prune(root, build);
                // start the app now and keep this window until its window is up (it takes a few seconds the
                // first time, while Windows checks the new files)
                form.BeginInvoke((Action)(() =>
                {
                    label.Text = "Starting Avatar Studio...";
                    bar.Style = ProgressBarStyle.Marquee;
                }));
                try
                {
                    app = start();
                    var waited = Stopwatch.StartNew();
                    while (waitWindow && waited.Elapsed.TotalSeconds < 60 && !app.HasExited)
                    {
                        app.Refresh();
                        if (app.MainWindowHandle != IntPtr.Zero) break;
                        Thread.Sleep(100);
                    }
                }
                catch (Exception ex) { startError = ex; }
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
        if (startError != null) throw startError;  // (Main shows it)
        started = app;
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
