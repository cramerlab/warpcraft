using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math.Optimization;
using CommandLine;
using Warp;
using Warp.Tools;

namespace Warp.Craft
{
    class Program
    {
        static void Main(string[] args)
        {
            int NIterations = 40;
            int NSteps = 20;
            int BatchSize = 128;
            int ThreadsPerGPU = 1;
            int MaxGPUs = 4;

            float ParticleDiameter = 350;
            float PixelSize = 2.5f;
            int OutputBFactor = -100;
            int FilterWindowSize = 30;
            float FSCThreshold = 0.7f;

            int DilateMask = 2;
            int BodyOverlap = 6;

            string InitCoarsePath = "it000_atomscoarse.mrc";// "NMA_15k_100.mrc";
            int NCoarseAtoms = 15000;
            int OversampleFineAtoms = 4;
            bool DoUpdateModes = false;
            int NModes = 15;

            int NMaxSegments = 15;
            int NSegmentsOptimization = 15;
            int NSegmentsReconstruction = 15;

            int NSamplesMC = 1;
            int NSamplesMCSignificant = 1;

            float Sigma2NMA = 1f;
            float Sigma2Params = 1f;

            bool PrereadParticles = true;

            string ProjDir = !Debugger.IsAttached ? Environment.CurrentDirectory : "";
            if (ProjDir[ProjDir.Length - 1] != '\\' && ProjDir[ProjDir.Length - 1] != '/')
                ProjDir = ProjDir + "\\";

            string ModelStarPath = ProjDir + "";
            string ForcedBodiesPath = "";

            if (!Debugger.IsAttached)
            {
                Options Options = new Options();
                if (Parser.Default.ParseArguments(args, Options))
                {
                    ModelStarPath = ProjDir + Options.ModelStarPath;
                    ParticleDiameter = Options.ParticleDiameter;
                    PixelSize = Options.PixelSize;
                    NIterations = Options.NIterations;
                    NSteps = Options.NSteps;
                    BatchSize = Options.BatchSize;
                    ThreadsPerGPU = Options.ThreadsPerGPU;
                    OutputBFactor = Options.BFactor;
                    FilterWindowSize = Options.WindowSize;
                    FSCThreshold = Options.FSCThreshold;

                    DilateMask = Options.DilateMask;
                    BodyOverlap = Options.BodyOverlap;

                    InitCoarsePath = Options.InitCoarsePath;
                    NCoarseAtoms = Options.NCoarseAtoms;
                    OversampleFineAtoms = Options.OversampleFine;
                    DoUpdateModes = Options.UpdateModes;
                    NModes = Options.NModes;

                    PrereadParticles = Options.PrereadParticles;
                }
                else
                {
                    return;
                }
            }

            Console.WriteLine("Executing with the following parameters:");
            Console.WriteLine($"  Input model: {ModelStarPath}");
            Console.WriteLine($"  Particle diameter: {ParticleDiameter} A");
            Console.WriteLine($"  Pixel size: {PixelSize} A");
            Console.WriteLine($"  Iterations: {NIterations}");
            Console.WriteLine($"  Steps per iteration: {NSteps}");
            Console.WriteLine($"  Batch size: {BatchSize}");
            Console.WriteLine($"  Threads per GPU: {ThreadsPerGPU}\n");
            Console.WriteLine($"  Pre-read particles: {PrereadParticles}\n");

            Console.WriteLine($"  Output B-factor: {OutputBFactor} A^2");
            Console.WriteLine($"  Local resolution window size: {FilterWindowSize} px");
            Console.WriteLine($"  Local resolution FSC threshold: {FSCThreshold}\n");

            Console.WriteLine($"  Dilate mask for fine model: {DilateMask} px");
            Console.WriteLine($"  Body overlap region: {BodyOverlap} px\n");

            Console.WriteLine($"  Init coarse atom model with: {InitCoarsePath}");
            Console.WriteLine($"  Number of coarse atoms: {NCoarseAtoms}");
            Console.WriteLine($"  Oversample fine atoms by: {OversampleFineAtoms}x");
            Console.WriteLine($"  Update modes after each iteration: {DoUpdateModes}");
            Console.WriteLine($"  Normal modes: {NModes}\n");

            if (!File.Exists(ModelStarPath))
            {
                Console.WriteLine("Could not find input model STAR file.");
                return;
            }

            Star TableMap = new Star(ModelStarPath, "map");
            Star TableBodies = new Star(ModelStarPath, "bodies");
            Star TableAtoms = Star.ContainsTable(ModelStarPath, "atoms") ? new Star(ModelStarPath, "atoms") : null; // Atoms are not necessarily defined for first iteration

            // If bodies are specified as a wildcard, find them
            {
                if (TableBodies.RowCount == 1 && TableBodies.GetRowValue(0, "wrpBodyMaskHard").Contains("*"))
                {
                    string Wildcard = TableBodies.GetRowValue(0, "wrpBodyMaskHard");
                    TableBodies = new Star(new [] {"wrpBodyMaskHard", "wrpBodyResolution"});

                    string WildcardFolder = Wildcard.Contains("/") ? Wildcard.Substring(0, Wildcard.LastIndexOf("/")) : "";
                    Wildcard = Wildcard.Substring(Wildcard.LastIndexOf("/") + 1);

                    foreach (var file in Directory.EnumerateFiles(ProjDir + WildcardFolder, Wildcard))
                    {
                        string MaskPath = file.Substring(ProjDir.Length).Replace("\\", "/");
                        TableBodies.AddRow(new List<string> { MaskPath, "100" });
                    }
                }
            }

            int NBodies = TableBodies.RowCount;

            int Size;   // Get box size from one of the reference maps
            {
                string MapPath = TableMap.GetRowValue(0, "wrpMapHalf1");

                Image Map = Image.FromFile(ProjDir + MapPath, new int2(1, 1), 0, typeof(float));
                Size = Map.Dims.X;
                Map.Dispose();
            }

            Console.WriteLine("Calculating masks.\n");

            #region Create global mask from the sum of all wide local masks
            Image MaskGlobal, MaskGlobalDilated;
            int MaskGlobalVoxels, MaskGlobalDilatedVoxels;
            //Projector ProjectorMask;
            {
                MaskGlobal = new Image(new int3(Size, Size, Size));

                Parallel.For(0, NBodies, new ParallelOptions { MaxDegreeOfParallelism = 2 }, r =>
                {
                    Image MaskLocal = Image.FromFile(ProjDir + TableBodies.GetRowValue(r, "wrpBodyMaskHard"), new int2(1, 1), 0, typeof (float));
                    lock (MaskGlobal)
                        MaskGlobal.Add(MaskLocal);
                    MaskLocal.Dispose();
                });

                MaskGlobal.Binarize(1e-5f);
                MaskGlobalDilated = MaskGlobal.AsDilatedMask(DilateMask);

                MaskGlobalVoxels = (int)(MaskGlobal.GetHostContinuousCopy().Sum() + 0.5f);
                MaskGlobalDilatedVoxels = (int)(MaskGlobalDilated.GetHostContinuousCopy().Sum() + 0.5f);

                MaskGlobal.FreeDevice();
                MaskGlobalDilated.FreeDevice();

                MaskGlobal.WriteMRC("d_maskglobal.mrc");
                MaskGlobalDilated.WriteMRC("d_maskglobaldilated.mrc");

                //Image MaskForParticles = MaskGlobal.AsDilatedMask(10);
                //ProjectorMask = new Projector(MaskForParticles, 2);
                //MaskForParticles.Dispose();
            }
            #endregion
            
            #region Calculate masks to insert each body into the frankenmap
            
            float[] BodySegmentation = new float[Size * Size * Size];
            {
                for (int i = 0; i < BodySegmentation.Length; i++)
                    BodySegmentation[i] = -1f;
                
                Parallel.For(0, NBodies, new ParallelOptions { MaxDegreeOfParallelism = 2 }, b =>
                {
                    Image HardMask = Image.FromFile(ProjDir + TableBodies.GetRowValue(b, "wrpBodyMaskHard"));
                    HardMask.Binarize(1e-6f);

                    float[] HardMaskData = HardMask.GetHostContinuousCopy();
                    lock (BodySegmentation)
                    {
                        Parallel.For(0, HardMaskData.Length, i =>
                        {
                            if (HardMaskData[i] > 0)
                                BodySegmentation[i] = b;
                        });
                    }

                    HardMask.Dispose();
                });
            }

            // If forced body masks are provided:
            float[] ForcedSegmentation = null;
            if (ForcedBodiesPath != "")
            {
                ForcedSegmentation = new float[Size * Size * Size];

                string Wildcard = ForcedBodiesPath;

                string WildcardFolder = Wildcard.Contains("/") ? Wildcard.Substring(0, Wildcard.LastIndexOf("/")) : "";
                Wildcard = Wildcard.Substring(Wildcard.LastIndexOf("/") + 1);

                int b = 0;
                foreach (var file in Directory.EnumerateFiles(ProjDir + WildcardFolder, Wildcard))
                {
                    Image HardMask = Image.FromFile(file);
                    HardMask.Binarize(1e-6f);

                    float[] HardMaskData = HardMask.GetHostContinuousCopy();
                    Parallel.For(0, HardMaskData.Length, i =>
                    {
                        if (HardMaskData[i] > 0)
                            ForcedSegmentation[i] = b;
                    });

                    HardMask.Dispose();
                    b++;
                }

                NSegmentsOptimization = b;
                NSegmentsReconstruction = b;
                NMaxSegments = b;
            }

            #endregion

            NMAMap AtomsCoarse;
            NMAMap[][] AtomsFineHalves = new NMAMap[GPU.GetDeviceCount()][];
            for (int i = 0; i < AtomsFineHalves.Length; i++)
                AtomsFineHalves[i] = new NMAMap[2];

            #region Pre-read particles into memory if needed

            Dictionary<string, float[]> ParticleStorage = new Dictionary<string, float[]>();
            if (PrereadParticles)
            {
                Console.WriteLine("Pre-reading particles:");

                #region Load particle parameters

                Star TableData = new Star(ProjDir + TableMap.GetRowValue(0, "wrpParticles"));

                List<int>[] SubsetRows = { new List<int>(), new List<int>() };
                for (int r = 0; r < TableData.RowCount; r++)
                    SubsetRows[(int)float.Parse(TableData.GetRowValue(r, "rlnRandomSubset")) - 1].Add(r);

                #endregion

                for (int h = 0, completed = 0; h < 2; h++)
                {
                    Parallel.For(0, SubsetRows[h].Count, r =>
                    {
                        int R = SubsetRows[h][r];

                        string ImageName = TableData.GetRowValue(R, "rlnImageName");

                        string[] AddressParts = ImageName.Split(new[] { "@" }, StringSplitOptions.RemoveEmptyEntries);
                        int Layer = int.Parse(AddressParts[0]) - 1;
                        string FileName = ProjDir + AddressParts[1];

                        // Get particle
                        Image Particle = Image.FromFile(FileName, new int2(Size, Size), 0, typeof (float), Layer);

                        lock (TableMap)
                        {
                            ParticleStorage.Add(ImageName, Particle.GetHostContinuousCopy());

                            completed++;
                            if (completed % 100 == 0 || completed == SubsetRows.Select(v => v.Count).Sum())
                            {
                                ClearCurrentConsoleLine();
                                Console.Write($"  {completed}/{SubsetRows.Select(v => v.Count).Sum()}");
                            }
                        }
                    });
                }

                Console.Write("\n\n");
            }

            #endregion

            for (int iter = 0; iter < NIterations; iter++)
            {
                Console.WriteLine($"Started iteration {iter + 1}:\n");

                if (iter > 0)   // In first iteration, tables are already loaded and possibly modified
                {
                    TableMap = new Star(ModelStarPath, "map");
                    TableBodies = new Star(ModelStarPath, "bodies");
                    TableAtoms = Star.ContainsTable(ModelStarPath, "atoms") ? new Star(ModelStarPath, "atoms") : null; // Atoms are not necessarily defined for first iteration
                }
                
                #region Load or make atom model
                {
                    if (InitCoarsePath == "" && (iter == 0 || DoUpdateModes)) // Initialize from scratch in first iteration, or update every iteration
                    {
                        Console.WriteLine("  Calculating normal modes – this can take a while.");

                        Image Frankenmap = Image.FromFile(ProjDir + TableMap.GetRowValue(0, "wrpFrankenmap"));
                        AtomsCoarse = NMAMap.FromScratch(Frankenmap, MaskGlobal, NCoarseAtoms, 100, 2.5f);
                        Frankenmap.Dispose();

                        AtomsCoarse.WriteToMRC(ProjDir + $"it000_atomscoarse.mrc");

                        if (NModes < AtomsCoarse.NModes)
                            AtomsCoarse.LimitModesTo(NModes);
                        AtomsCoarse.SmoothModes(128);
                    }
                    else if (InitCoarsePath != "" && iter == 0) // Initialize with precalculated model
                    {
                        Console.WriteLine($"  Initializing normal modes with {InitCoarsePath}.");

                        AtomsCoarse = new NMAMap(ProjDir + InitCoarsePath);
                        if (NModes < AtomsCoarse.NModes)
                            AtomsCoarse.LimitModesTo(NModes);
                        NModes = AtomsCoarse.NModes;
                        NCoarseAtoms = AtomsCoarse.NAtoms;

                        AtomsCoarse.SmoothModes(128);
                    }
                    else if (TableAtoms != null) // Take model from previous iteration
                    {
                        Console.WriteLine("  Using normal modes from previous iteration.");

                        AtomsCoarse = new NMAMap(ProjDir + TableAtoms.GetRowValue(0, "wrpAtomsCoarse"));
                    }
                    else
                    {
                        Console.WriteLine("  Atom model initialization went wrong: after first iteration, model.star should contain a path to the coarse model.");
                        return;
                    }

                    AtomsCoarse.AddSegmentation(new Image(BodySegmentation, new int3(Size, Size, Size)));

                    // Store coarse atoms for this iteration in a new file, regardless of whether the model is new or old
                    AtomsCoarse.WriteToMRC(ProjDir + $"it{iter + 1:D3}_atomscoarse.mrc");
                    if (TableAtoms == null)
                    {
                        TableAtoms = new Star(new[] { "wrpAtomsCoarse" });
                        TableAtoms.AddRow(new List<string> { $"it{(iter + 1):D3}_atomscoarse.mrc" });
                    }
                    else
                    {
                        if (!TableAtoms.HasColumn("wrpAtomsCoarse"))
                            TableAtoms.AddColumn("wrpAtomsCoarse");
                        TableAtoms.SetRowValue(0, "wrpAtomsCoarse", $"it{(iter + 1):D3}_atomscoarse.mrc");
                    }

                    AtomsCoarse.RasterizeInVolume(new int3(Size, Size, Size)).WriteMRC("d_atomscoarse_groundmode.mrc");
                    for (int m = 0; m < AtomsCoarse.NModes; m++)
                    {
                        float[] Weights = new float[AtomsCoarse.NModes];
                        Weights[m] = 2;
                        AtomsCoarse.RasterizeDeformedInVolume(new int3(Size, Size, Size), Weights).WriteMRC($"d_atomscoarse_mode{m:D2}.mrc");
                    }

                    AtomsCoarse.AddSegmentation(new Image(BodySegmentation, new int3(Size, Size, Size)));

                    Console.WriteLine("");
                }
                #endregion

                //AtomsCoarse.ReduceToNSegments(20);
                //Image[] TestMasks = AtomsCoarse.GetSoftSegmentMasks(BodyOverlap, BodyOverlap * 1.5f);
                //for (int i = 0; i < TestMasks.Length; i++)
                //{
                //    TestMasks[i].WriteMRC($"d_testmask{i:D2}.mrc");
                //}
                //float[][] TestMasksData = TestMasks.Select(v => v.GetHostContinuousCopy()).ToArray();
                //float[] TestMaskSum = new float[TestMasksData[0].Length];
                //for (int i = 0; i < TestMaskSum.Length; i++)
                //    for (int j = 0; j < AtomsCoarse.NSegments; j++)
                //        TestMaskSum[i] += TestMasksData[j][i];
                //new Image(TestMaskSum, AtomsCoarse.DimsVolume).WriteMRC("d_testmasksum.mrc");

                int SizeCoarse;
                float ScaleFactor;

                #region Estimate current per-body resolution, construct fine atom models from locally filtered maps

                int BestResolutionShell = 0;
                float[] InitialResolution = new float[NBodies];
                int[] BodyMaxShell = new int[NBodies];
                float3[] BodyCenter = new float3[NBodies];
                Projector[] DebugProjectors = new Projector[2];
                {
                    Console.WriteLine("  Estimating per-body resolution:");
                    Image Map1 = Image.FromFile(ProjDir + TableMap.GetRowValue(0, "wrpMapHalf1"), new int2(1, 1), 0, typeof (float));
                    Image Map2 = Image.FromFile(ProjDir + TableMap.GetRowValue(0, "wrpMapHalf2"), new int2(1, 1), 0, typeof (float));

                    #region Estimate local resolution
                    {
                        Image LocalRes = new Image(IntPtr.Zero, new int3(Size, Size, Size));
                        Image Map1Pristine = Map1;
                        Image Map2Pristine = Map2;

                        if (iter > 0)
                        {
                            Map1Pristine = Image.FromFile(ProjDir + TableMap.GetRowValue(0, "wrpMapHalfPristine1"), new int2(1, 1), 0, typeof(float));
                            Map2Pristine = Image.FromFile(ProjDir + TableMap.GetRowValue(0, "wrpMapHalfPristine2"), new int2(1, 1), 0, typeof(float));
                        }

                        GPU.LocalRes(Map1Pristine.GetDevice(Intent.Read),
                                     Map2Pristine.GetDevice(Intent.Read),
                                     new int3(Size, Size, Size),
                                     PixelSize,
                                     IntPtr.Zero,
                                     IntPtr.Zero,
                                     LocalRes.GetDevice(Intent.Write),
                                     IntPtr.Zero,
                                     FilterWindowSize,
                                     FSCThreshold,
                                     false,
                                     10,
                                     0,
                                     0,
                                     false,
                                     false);

                        if (iter > 0)
                        {
                            Map1Pristine.Dispose();
                            Map2Pristine.Dispose();
                        }

                        float[] LocalResData = LocalRes.GetHostContinuousCopy().Select(v => Size * PixelSize / Math.Max(1, v)).ToArray();   // LocalResData now as frequencies

                        float GlobalAverageShell = MathHelper.MeanWeighted(LocalResData, MaskGlobal.GetHostContinuousCopy());
                        BestResolutionShell = (int)GlobalAverageShell;
                        
                        LocalRes.WriteMRC("d_localres.mrc");

                        //Image LocalRes = Image.FromFile("d_localres.mrc");

                        Image[] MasksHard = AtomsCoarse.GetSegmentMasks();
                        int Done = 0;
                        Parallel.For(0, NBodies, b =>
                        {
                            Image MaskHard = MasksHard[b];
                            BodyCenter[b] = MaskHard.AsCenterOfMass();
                            float[] MaskHardData = MaskHard.GetHostContinuousCopy();

                            float AverageShell = MathHelper.MeanWeighted(LocalResData, MaskHardData);

                            BodyMaxShell[b] = (int)AverageShell;
                            InitialResolution[b] = AverageShell;    // Frequency in global volume frame

                            MaskHard.Dispose();

                            lock (TableAtoms)
                            {
                                //BestResolutionShell = Math.Max((int)AverageShell, BestResolutionShell);
                                ClearCurrentConsoleLine();
                                Console.Write($"    {++Done}/{NBodies}");
                            }
                        });

                        LocalRes.Dispose();
                    }
                    #endregion
                    Console.Write("\n");

                    #region Use the Frankenmaps to calculate per-voxel resolution based on the estimated per-body resolutions
                    
                    Image[] MasksSmooth = AtomsCoarse.GetSoftSegmentMasks(BodyOverlap, BodyOverlap * 1.5f);

                    Image BodyResolution, BodyBFactor;
                    {
                        float[] BodyResolutionData = new float[Size * Size * Size];

                        for (int b = 0; b < MasksSmooth.Length; b++)
                        {
                            float ResVal = InitialResolution[b];
                            float[] FrankenMaskData = MasksSmooth[b].GetHostContinuousCopy();

                            for (int i = 0; i < BodyResolutionData.Length; i++)
                                BodyResolutionData[i] += ResVal * FrankenMaskData[i];
                        }

                        for (int i = 0; i < BodyResolutionData.Length; i++)
                            BodyResolutionData[i] = Math.Min(Size * PixelSize / Math.Max(Math.Min(BestResolutionShell, BodyResolutionData[i]), 1f), FilterWindowSize / 2 * PixelSize);

                        BodyResolution = new Image(BodyResolutionData, new int3(Size, Size, Size));
                        BodyBFactor = new Image(new int3(Size, Size, Size));

                        BodyResolution.WriteMRC("d_bodyresolution.mrc");
                    }

                    foreach (var mask in MasksSmooth)
                        mask.Dispose();

                    #endregion

                    #region Filter both half-maps to per-voxel resolution derived from bodies

                    GPU.LocalFilter(Map1.GetDevice(Intent.Read),
                                    Map1.GetDevice(Intent.Write),
                                    new int3(Size, Size, Size),
                                    PixelSize,
                                    BodyResolution.GetDevice(Intent.Read),
                                    BodyBFactor.GetDevice(Intent.Read),
                                    FilterWindowSize,
                                    0f);

                    GPU.LocalFilter(Map2.GetDevice(Intent.Read),
                                    Map2.GetDevice(Intent.Write),
                                    new int3(Size, Size, Size),
                                    PixelSize,
                                    BodyResolution.GetDevice(Intent.Read),
                                    BodyBFactor.GetDevice(Intent.Read),
                                    FilterWindowSize,
                                    0f);

                    //Map1 = Image.FromFile("d_map1.mrc");
                    //Map2 = Image.FromFile("d_map2.mrc");
                    Map1.WriteMRC("d_map1.mrc");
                    Map2.WriteMRC("d_map2.mrc");

                    BodyResolution.Dispose();
                    BodyBFactor.Dispose();

                    #endregion

                    //Image SoftMask = Image.FromFile(ProjDir + TableMap.GetRowValue(0, "wrpSoftMask"));
                    //Map1.Multiply(SoftMask);
                    //Map2.Multiply(SoftMask);
                    //SoftMask.Dispose();

                    Map1.FreeDevice();
                    Map2.FreeDevice();

                    //DebugProjectors[0] = new Projector(Map1, 2);
                    //DebugProjectors[1] = new Projector(Map2, 2);

                    // Set maximum image size and scale factor to match current average resolution
                    SizeCoarse = BestResolutionShell * 2;
                    ScaleFactor = (float)SizeCoarse / Size;

                    Console.WriteLine($"  Considering data until {Size * PixelSize / BestResolutionShell:F2} A.");
                    if (BestResolutionShell < 3)
                    {
                        Console.WriteLine("  ERROR: At least one of the bodies cannot be resolved at all. Aborting.");
                        return;
                    }

                    if (iter == 0)
                    {
                        Console.WriteLine($"  Initial resolution for individual bodies at {FSCThreshold:F3} cutoff is:");
                        for (int b = 0; b < NBodies; b++)
                            Console.WriteLine($"    {b + 1}: {Size * PixelSize / InitialResolution[b]:F2} A");

                        Console.Write("\n");
                    }

                    #region Construct fine atom models

                    Console.WriteLine("  Calculating fine NMA models.");

                    NMAMap AtomsForFine = AtomsCoarse.GetCopy();

                    if (ForcedSegmentation == null)
                        AtomsForFine.ReduceToNSegments(NSegmentsOptimization);
                    else
                        AtomsForFine.AddSegmentation(new Image(ForcedSegmentation, new int3(Size, Size, Size)));

                    //int gpuID = 0;
                    Helper.ForEachGPUOnce(gpuID =>
                    {
                        float SigmaFine, CorrFine;
                        Image Map1Copy = new Image(Map1.GetHost(Intent.Read), Map1.Dims);
                        Image Map2Copy = new Image(Map2.GetHost(Intent.Read), Map2.Dims);

                        AtomsFineHalves[gpuID][0] = AtomsForFine.GetCopy();
                        AtomsFineHalves[gpuID][0].InitializeBodyProjectors(BodyOverlap, BodyOverlap * 1.5f, Map1Copy, SizeCoarse, 2);
                        AtomsFineHalves[gpuID][0].FreeOnDevice();

                        AtomsFineHalves[gpuID][1] = AtomsForFine.GetCopy();
                        AtomsFineHalves[gpuID][1].InitializeBodyProjectors(BodyOverlap, BodyOverlap * 1.5f, Map2Copy, SizeCoarse, 2);
                        AtomsFineHalves[gpuID][1].FreeOnDevice();

                        if (gpuID == 0)
                        {
                            AtomsFineHalves[0][0].WriteToMRC("d_atomsfine1.mrc");
                            AtomsFineHalves[0][1].WriteToMRC("d_atomsfine2.mrc");

                            AtomsFineHalves[0][0].RasterizeInVolume(new int3(Size, Size, Size)).WriteMRC("d_atomsfine.mrc");
                            for (int m = 0; m < AtomsFineHalves[0][0].NModes; m++)
                            {
                                float[] Weights = new float[AtomsFineHalves[0][0].NModes];
                                Weights[m] = 2;
                                AtomsFineHalves[0][0].RasterizeDeformedInVolume(new int3(Size, Size, Size), Weights).WriteMRC($"d_atomsfine_mode{m:D2}.mrc");
                            }
                        }
                    }, MaxGPUs);

                    Map1.Dispose();
                    Map2.Dispose();

                    #endregion
                }
                #endregion

                #region Load particle parameters

                Star TableData = new Star(ProjDir + TableMap.GetRowValue(0, "wrpParticles"));
                int NParticles = TableData.RowCount;

                List<int>[] SubsetRows = { new List<int>(), new List<int>() };
                for (int r = 0; r < TableData.RowCount; r++)
                    SubsetRows[(int)float.Parse(TableData.GetRowValue(r, "rlnRandomSubset")) - 1].Add(r);

                Dictionary<string, int> GroupMapping = new Dictionary<string, int>();
                float3[][] ParticleAngles = { new float3[SubsetRows[0].Count], new float3[SubsetRows[1].Count] };
                float2[][] ParticleShifts = { new float2[SubsetRows[0].Count], new float2[SubsetRows[1].Count] };
                float[][] ParticleNMAWeights = { new float[SubsetRows[0].Count * NModes], new float[SubsetRows[1].Count * NModes] };

                CTFStruct[][] ParticleCTFParams = { new CTFStruct[SubsetRows[0].Count], new CTFStruct[SubsetRows[1].Count] };
                int[][] ParticleGroups = { new int[SubsetRows[0].Count], new int[SubsetRows[1].Count] };

                float3[][][] MCAngles = { new float3[SubsetRows[0].Count][], new float3[SubsetRows[1].Count][] };
                float2[][][] MCShifts = { new float2[SubsetRows[0].Count][], new float2[SubsetRows[1].Count][] };
                float[][][] MCNMAWeights = { new float[SubsetRows[0].Count][], new float[SubsetRows[1].Count][] };
                float[][][] MCScores = { new float[SubsetRows[0].Count][], new float[SubsetRows[1].Count][] };

                for (int h = 0; h < 2; h++)
                {
                    // Get CTF parameters
                    for (int r = 0; r < SubsetRows[h].Count; r++)
                    {
                        int R = SubsetRows[h][r];

                        float Voltage = TableData.GetRowValueFloat(R, "rlnVoltage");
                        float DefocusU = TableData.GetRowValueFloat(R, "rlnDefocusU") / 1e4f;
                        float DefocusV = TableData.GetRowValueFloat(R, "rlnDefocusV") / 1e4f;
                        float DefocusAngle = TableData.GetRowValueFloat(R, "rlnDefocusAngle");
                        float Cs = TableData.GetRowValueFloat(R, "rlnSphericalAberration");
                        //float Phase = TableData[0].GetRowValueFloat(R, "rlnPhaseShift");
                        float Contrast = TableData.GetRowValueFloat(R, "rlnAmplitudeContrast");

                        CTF ParticleCTF = new CTF
                        {
                            PixelSize = (decimal)PixelSize,
                            Voltage = (decimal)Voltage,
                            Defocus = (decimal)(DefocusU + DefocusV) * 0.5M,
                            DefocusDelta = (decimal)(DefocusU - DefocusV),
                            DefocusAngle = (decimal)DefocusAngle,
                            Cs = (decimal)Cs,
                            //PhaseShift = (decimal)Phase / 180M,
                            Amplitude = (decimal)Contrast
                        };

                        ParticleCTFParams[h][r] = ParticleCTF.ToStruct();
                    }
                    
                    // Get rotations and shifts
                    for (int r = 0; r < SubsetRows[h].Count; r++)
                    {
                        int R = SubsetRows[h][r];

                        ParticleAngles[h][r] = new float3(TableData.GetRowValueFloat(R, "rlnAngleRot"),
                                                          TableData.GetRowValueFloat(R, "rlnAngleTilt"),
                                                          TableData.GetRowValueFloat(R, "rlnAnglePsi"));

                        ParticleShifts[h][r] = new float2(TableData.GetRowValueFloat(R, "rlnOriginX"),
                                                          TableData.GetRowValueFloat(R, "rlnOriginY"));
                    }

                    // Get NMA weights if there are any
                    {
                        string[] NMAColumns = TableData.GetColumnNames().Where(v => v.Contains("wrpNMAWeight")).ToArray();
                        bool AllColumnsPresent = true;
                        for (int i = 1; i <= NModes; i++)
                            if (!NMAColumns.Contains($"wrpNMAWeight{i}"))
                                AllColumnsPresent = false;

                        if (AllColumnsPresent)
                        {
                            for (int m = 0; m < NModes; m++)
                                for (int r = 0; r < SubsetRows[h].Count; r++)
                                {
                                    int R = SubsetRows[h][r];

                                    ParticleNMAWeights[h][r * NModes + m] = TableData.GetRowValueFloat(SubsetRows[h][r], $"wrpNMAWeight{m + 1}");
                                }
                        }
                    }

                    // Figure out groups
                    for (int r = 0; r < SubsetRows[h].Count; r++)
                    {
                        int R = SubsetRows[h][r];

                        string MicName = TableData.HasColumn("rlnGroupName") ? TableData.GetRowValue(R, "rlnGroupName") :
                                                                               TableData.GetRowValue(R, "rlnMicrographName");
                        MicName = MicName.Substring(MicName.LastIndexOf("/") + 1);

                        if (!GroupMapping.ContainsKey(MicName))
                            GroupMapping.Add(MicName, GroupMapping.Count);

                        ParticleGroups[h][r] = GroupMapping[MicName];
                    }
                }

                // Map particles to groups
                int NGroups = GroupMapping.Count;
                List<int>[][] GroupParticles = { new List<int>[NGroups], new List<int>[NGroups] };
                for (int h = 0; h < 2; h++)
                    for (int g = 0; g < NGroups; g++)
                        GroupParticles[h][g] = new List<int>();

                for (int h = 0; h < 2; h++)
                    for (int p = 0; p < SubsetRows[h].Count; p++)
                        GroupParticles[h][ParticleGroups[h][p]].Add(p);

                #endregion

                #region Noise and scale estimation

                Console.WriteLine("  Estimating noise spectra for each micrograph:");

                //Image Dummy = Image.FromFile("E:\\multibody2\\tf2h\\it040_frankenmap_half1.mrc");
                //Projector DummyProj = new Projector(Dummy, 2);

                float[][] GroupInvNoise = new float[NGroups][];
                float[] GroupScale = new float[NGroups].Select(v => 1f).ToArray();
                float[] ParticleNormalization = new float[NParticles].Select(v => 1f).ToArray();
                {
                    int[] PixelShell = new int[(SizeCoarse / 2 + 1) * SizeCoarse];
                    for (int y = 0; y < SizeCoarse; y++)
                        for (int x = 0; x < SizeCoarse / 2 + 1; x++)
                        {
                            int xx = x;
                            int yy = y < SizeCoarse / 2 + 1 ? y : y - SizeCoarse;

                            int R = (int)Math.Round(Math.Sqrt(xx * xx + yy * yy));
                            PixelShell[y * (SizeCoarse / 2 + 1) + x] = R < SizeCoarse / 2 ? R : -1;
                        }

                    #region Iterate over groups

                    if (true)
                    {
                        int GroupsDone = 0;
                        Helper.ForGPU(0, NGroups, (g, gpuID) =>
                        {
                            float[] Noise1D = new float[SizeCoarse / 2];
                            int[] Samples1D = new int[SizeCoarse / 2];
                            float SumPart = 0, SumProj = 0;

                            Image CTFCoordsCoarse = CTF.GetCTFCoords(SizeCoarse, Size);

                            #region Create mask

                            Image ParticleMask = new Image(new int3(SizeCoarse, SizeCoarse, 1));
                            {
                                float[] ParticleMaskData = ParticleMask.GetHost(Intent.Write)[0];
                                float Radius2 = ParticleDiameter / 2 / PixelSize * ScaleFactor;

                                for (int y = 0; y < SizeCoarse; y++)
                                {
                                    int yy = y - SizeCoarse / 2;
                                    yy *= yy;

                                    for (int x = 0; x < SizeCoarse; x++)
                                    {
                                        int xx = x - SizeCoarse / 2;
                                        xx *= xx;
                                        float R = (float)Math.Sqrt(xx + yy);

                                        float V = R <= Radius2 ? 1f : (float)(Math.Cos(Math.Min(1, (R - Radius2) / 10) * Math.PI) * 0.5 + 0.5);
                                        ParticleMaskData[y * SizeCoarse + x] = V;
                                    }
                                }
                            }

                            #endregion

                            float2[][][] ParticleData = new float2[2][][];
                            float2[][][] ProjData = new float2[2][][];

                            for (int h = 0; h < 2; h++)
                            {
                                int NParts = GroupParticles[h][g].Count;
                                if (NParts == 0)
                                    continue;

                                #region Get particle parameters

                                float3[] GroupAngles = new float3[NParts];
                                float2[] GroupShifts = new float2[NParts];
                                CTFStruct[] GroupCTFParams = new CTFStruct[NParts];
                                float[][] GroupNMAWeights = Helper.ArrayOfFunction(() => new float[NModes], NParts);

                                for (int p = 0; p < NParts; p++)
                                {
                                    int pi = GroupParticles[h][g][p];

                                    GroupAngles[p] = ParticleAngles[h][pi];
                                    GroupShifts[p] = ParticleShifts[h][pi];
                                    GroupCTFParams[p] = ParticleCTFParams[h][pi];
                                    for (int m = 0; m < NModes; m++)
                                        GroupNMAWeights[p][m] = ParticleNMAWeights[h][pi * NModes + m];
                                }

                                #endregion

                                #region Create CTF

                                Image GroupCTF;
                                {
                                    GroupCTF = new Image(new int3(SizeCoarse, SizeCoarse, NParts), true);
                                    GPU.CreateCTF(GroupCTF.GetDevice(Intent.Write),
                                                  CTFCoordsCoarse.GetDevice(Intent.Read),
                                                  (uint)CTFCoordsCoarse.ElementsComplex,
                                                  GroupCTFParams,
                                                  false,
                                                  (uint)NParts);
                                    //GroupCTF.WriteMRC("d_groupctf.mrc");
                                }

                                #endregion

                                #region Project reference

                                Image ProjectionsFT = AtomsFineHalves[gpuID][h].ProjectBodies(new int2(SizeCoarse, SizeCoarse),
                                                                                              GroupNMAWeights,
                                                                                              GroupAngles.Select(a => a * Helper.ToRad).ToArray(),
                                                                                              new float2[NParts],
                                                                                              Helper.ArrayOfFunction(() => Helper.ArrayOfConstant(1f, AtomsFineHalves[gpuID][h].NSegments), NParts),
                                                                                              NParts);
                                //Image ProjectionsFT = DebugProjectors[h].Project(new int2(SizeCoarse, SizeCoarse), GroupAngles.Select(a => a * Helper.ToRad).ToArray(), (int)(192 * 2.74f / 15f));
                                //Image ProjectionsFT = new Image(new int3(SizeCoarse, SizeCoarse, NParts), true, true);
                                AtomsFineHalves[gpuID][h].FreeOnDevice();

                                #endregion

                                ProjectionsFT.Multiply(GroupCTF);
                                //ProjectionsFT.Multiply(1f / Size);

                                #region Load particles

                                Image ParticlesCoarseFT;
                                {
                                    Image ParticlesRaw = new Image(new int3(Size, Size, NParts));
                                    float[][] ParticlesRawData = ParticlesRaw.GetHost(Intent.Write);

                                    for (int p = 0; p < NParts; p++)
                                    {
                                        int pi = GroupParticles[h][g][p];
                                        string ImageName = TableData.GetRowValue(SubsetRows[h][pi], "rlnImageName");

                                        if (!PrereadParticles)
                                        {
                                            string[] AddressParts = ImageName.Split(new[] { "@" }, StringSplitOptions.RemoveEmptyEntries);
                                            int Layer = int.Parse(AddressParts[0]) - 1;
                                            string FileName = ProjDir + AddressParts[1];

                                            Image Particle = Image.FromFile(FileName, new int2(Size, Size), 0, typeof (float), Layer);
                                            ParticlesRawData[p] = Particle.GetHostContinuousCopy();
                                            Particle.Dispose();
                                        }
                                        else
                                        {
                                            ParticlesRawData[p] = ParticleStorage[ImageName].ToArray();
                                        }
                                    }

                                    ParticlesRaw.ShiftSlices(GroupShifts.Select(v => new float3(v.X, v.Y, 0)).ToArray());

                                    Image ParticlesCoarse = ParticlesRaw.AsScaled(new int2(SizeCoarse, SizeCoarse));
                                    ParticlesCoarse.Multiply(1f / (SizeCoarse * SizeCoarse));
                                    ParticlesRaw.Dispose();
                                    //ParticlesCoarse.WriteMRC("d_particlescoarse.mrc");

                                    ParticlesCoarse.MultiplySlices(ParticleMask);
                                    ParticlesCoarse.RemapToFT();

                                    ParticlesCoarseFT = ParticlesCoarse.AsFFT();
                                    ParticlesCoarse.Dispose();
                                }

                                #endregion

                                //ParticlesCoarseFT.Subtract(ProjectionsFT);

                                #region Calculate squared differences for variance estimation

                                ParticleData[h] = ParticlesCoarseFT.GetHostComplexCopy();
                                ProjData[h] = ProjectionsFT.GetHostComplexCopy();
                                for (int p = 0; p < NParts; p++)
                                {
                                    float2[] PartSlice = ParticleData[h][p];
                                    float2[] ProjSlice = ProjData[h][p];
                                    for (int i = 0; i < PixelShell.Length; i++)
                                        if (PixelShell[i] >= 0)
                                        {
                                            SumPart += PartSlice[i].X * ProjSlice[i].X + PartSlice[i].Y * ProjSlice[i].Y;
                                            SumProj += ProjSlice[i].X * ProjSlice[i].X + ProjSlice[i].Y * ProjSlice[i].Y;
                                        }
                                }

                                #endregion

                                ParticlesCoarseFT.Dispose();
                                ProjectionsFT.Dispose();
                                GroupCTF.Dispose();
                            }

                            ParticleMask.Dispose();
                            CTFCoordsCoarse.Dispose();

                            GroupScale[g] = SumPart / Math.Max(1e-20f, SumProj);

                            for (int h = 0; h < 2; h++)
                            {
                                int NParts = GroupParticles[h][g].Count;
                                if (NParts == 0)
                                    continue;

                                for (int p = 0; p < NParts; p++)
                                {
                                    int pi = SubsetRows[h][GroupParticles[h][g][p]];
                                    float2[] PartSlice = ParticleData[h][p];
                                    float2[] ProjSlice = ProjData[h][p];
                                    for (int i = 0; i < PixelShell.Length; i++)
                                        if (PixelShell[i] >= 0)
                                        {
                                            float2 Diff = PartSlice[i] - ProjSlice[i] * GroupScale[g];
                                            Noise1D[PixelShell[i]] += Diff.X * Diff.X + Diff.Y * Diff.Y;
                                            Samples1D[PixelShell[i]]++;

                                            ParticleNormalization[pi] += Diff.X * Diff.X + Diff.Y * Diff.Y;
                                        }

                                    ParticleNormalization[pi] = (float)Math.Sqrt(ParticleNormalization[pi] * 2);
                                }
                            }

                            for (int r = 0; r < SizeCoarse / 2; r++)
                                Noise1D[r] /= Samples1D[r] * 2;

                            float[] InvNoise = new float[PixelShell.Length];
                            for (int i = 0; i < InvNoise.Length; i++)
                                if (PixelShell[i] >= 0)
                                    InvNoise[i] = 1f / Noise1D[PixelShell[i]];

                            GroupInvNoise[g] = InvNoise;

                            lock (TableData)
                            {
                                ClearCurrentConsoleLine();
                                Console.Write($"    {++GroupsDone}/{NGroups}");
                            }
                        }, 1, MaxGPUs);
                    }

                    #endregion

                    Console.Write("\n");
                }

                // Normalize particle norm factors to average to 1.0
                {
                    float AverageNorm = MathHelper.Mean(ParticleNormalization);
                    ParticleNormalization = ParticleNormalization.Select(v => v / AverageNorm).ToArray();
                }

                // Normalize group scales to have average of 1.0
                {
                    float AverageScale = MathHelper.Mean(GroupScale);
                    GroupScale = GroupScale.Select(v => v / AverageScale).ToArray();
                }

                new Image(GroupInvNoise, new int3(SizeCoarse, SizeCoarse, NGroups), true).WriteMRC("d_invsigma2.mrc");
                //GroupInvNoise = Image.FromFile("d_invsigma2.mrc").GetHost(Intent.Read);

                #endregion

                //RandomNormal RandN = new RandomNormal(123);
                //ParticleNMAWeights = new[]
                //{
                //    Helper.ArrayOfFunction(() => RandN.NextSingle(0, 0.5f), SubsetRows[0].Count * NModes),
                //    Helper.ArrayOfFunction(() => RandN.NextSingle(0, 0.5f), SubsetRows[1].Count * NModes)
                //};

                #region Accuracy estimation

                Console.WriteLine("  Estimating parameter accuracy");
                float[] ParameterAccuracy = new float[5 + NModes];
                {
                    int NParams = ParameterAccuracy.Length;
                    List<float>[] AccuracyValues = Helper.ArrayOfFunction(() => new List<float>(), NParams);

                    for (int h = 0, completed = 0; h < 2; h++)
                    {
                        List<int> BatchStarts = new List<int>();
                        for (int batchStart = 0; batchStart < SubsetRows[h].Count; batchStart += BatchSize)
                            BatchStarts.Add(batchStart);

                        Helper.ForEachGPU(BatchStarts, (batchStart, gpuID) =>
                        {
                            Image CTFCoordsCoarse = CTF.GetCTFCoords(SizeCoarse, Size);

                            #region Generate per-particle 2D CTF

                            Image ParticlesCTF;
                            {
                                ParticlesCTF = new Image(new int3(SizeCoarse, SizeCoarse, NParams), true);
                                GPU.CreateCTF(ParticlesCTF.GetDevice(Intent.Write),
                                              CTFCoordsCoarse.GetDevice(Intent.Read),
                                              (uint)CTFCoordsCoarse.ElementsComplex,
                                              Helper.ArrayOfConstant(ParticleCTFParams[h][batchStart], NParams),
                                              false,
                                              (uint)NParams);

                                float[][] ParticlesCTFData = ParticlesCTF.GetHost(Intent.ReadWrite);
                                for (int p = 0; p < NParams; p++)
                                {
                                    float Scale = GroupScale[ParticleGroups[h][batchStart]];
                                    for (int i = 0; i < ParticlesCTFData[p].Length; i++)
                                        ParticlesCTFData[p][i] *= Scale;
                                }
                                //ParticlesCTF.WriteMRC("d_particlectf.mrc");
                            }
                            CTFCoordsCoarse.Dispose();

                            #endregion

                            #region Generate per-particle 2D inverse noise spectrum
                            Image ParticlesInvSigma2;
                            {
                                float[][] InvSigma2Data = new float[NParams][];
                                for (int p = 0; p < NParams; p++)
                                    InvSigma2Data[p] = GroupInvNoise[ParticleGroups[h][batchStart]].ToArray();

                                ParticlesInvSigma2 = new Image(InvSigma2Data, new int3(SizeCoarse, SizeCoarse, NParams), true);
                            }
                            #endregion

                            float2 StartShift = ParticleShifts[h][batchStart];
                            float3 StartAngle = ParticleAngles[h][batchStart];
                            float[] StartNMAWeights = ParticleNMAWeights[h].Skip(batchStart * NModes).Take(NModes).ToArray();

                            Image StartProjFT = AtomsFineHalves[gpuID][h].ProjectBodies(new int2(SizeCoarse, SizeCoarse),
                                                                                        Helper.ArrayOfConstant(StartNMAWeights, NParams),
                                                                                        Helper.ArrayOfConstant(StartAngle, NParams).Select(v => v * Helper.ToRad).ToArray(),
                                                                                        Helper.ArrayOfConstant(StartShift, NParams).Select(v => -v).ToArray(),
                                                                                        Helper.ArrayOfConstant(Helper.ArrayOfConstant(1f, AtomsFineHalves[gpuID][h].NSegments), NParams),
                                                                                        NParams);
                            //StartProj.WriteMRC("d_startproj.mrc");
                            StartProjFT.Multiply(ParticlesCTF);

                            float[] ExpSnr = new float[NParams];
                            float[] ParamChange = new float[NParams];
                            bool StillChanging = true;
                            int Tries = 0;
                            while (StillChanging)
                            {
                                StillChanging = false;

                                for (int p = 0; p < NParams; p++)
                                    if (ExpSnr[p] < 4.61f) // exp(-4.61) = 0.01
                                    {
                                        StillChanging = true;

                                        if (ParamChange[p] < 0.2f)
                                            ParamChange[p] += 0.05f;
                                        else if (ParamChange[p] < 1f)
                                            ParamChange[p] += 0.1f;
                                        else if (ParamChange[p] < 2f)
                                            ParamChange[p] += 0.2f;
                                        else if (ParamChange[p] < 5f)
                                            ParamChange[p] += 0.5f;
                                        else if (ParamChange[p] < 10f)
                                            ParamChange[p] += 1.0f;
                                        else if (ParamChange[p] < 20f)
                                            ParamChange[p] += 2f;
                                        else
                                            ParamChange[p] += 5f;
                                    }

                                if (!StillChanging || Tries++ > 30)
                                    break;

                                float2[] ChangedShifts = Helper.ArrayOfConstant(StartShift, NParams);
                                float3[] ChangedAngles = Helper.ArrayOfConstant(StartAngle, NParams);
                                float[][] ChangedNMAWeights = Helper.ArrayOfFunction(() => StartNMAWeights.ToArray(), NParams);

                                ChangedShifts[0].X += ParamChange[0];
                                ChangedShifts[1].Y += ParamChange[1];
                                ChangedAngles[2].X += ParamChange[2];
                                ChangedAngles[3].Y += ParamChange[3];
                                ChangedAngles[4].Z += ParamChange[4];
                                for (int p = 5; p < NParams; p++)
                                    ChangedNMAWeights[p][p - 5] += ParamChange[p];

                                Image ChangedProjFT = AtomsFineHalves[gpuID][h].ProjectBodies(new int2(SizeCoarse, SizeCoarse),
                                                                                              ChangedNMAWeights,
                                                                                              ChangedAngles.Select(v => v * Helper.ToRad).ToArray(),
                                                                                              ChangedShifts.Select(v => -v).ToArray(),
                                                                                              Helper.ArrayOfConstant(Helper.ArrayOfConstant(1f, AtomsFineHalves[gpuID][h].NSegments), NParams),
                                                                                              NParams);
                                //ChangedProj.WriteMRC("d_changedproj.mrc");

                                float[] Diff2 = new float[NParams];
                                GPU.ParticleNMAGetDiff(StartProjFT.GetDevice(Intent.Read),
                                                       ChangedProjFT.GetDevice(Intent.Read),
                                                       ParticlesCTF.GetDevice(Intent.Read),
                                                       ParticlesInvSigma2.GetDevice(Intent.Read),
                                                       new int2(SizeCoarse, SizeCoarse),
                                                       Diff2,
                                                       (uint)NParams);
                                ChangedProjFT.Dispose();

                                ExpSnr = Diff2;
                            }

                            ParticlesInvSigma2.Dispose();
                            ParticlesCTF.Dispose();
                            StartProjFT.Dispose();

                            for (int p = 0; p < NParams; p++)
                                AccuracyValues[p].Add(ParamChange[p]);
                            
                            lock (TableData)
                            {
                                ClearCurrentConsoleLine();
                                Console.Write($"    {++completed}/{BatchStarts.Count}");
                            }
                        }, 1, MaxGPUs);

                        Helper.ForEachGPUOnce(gpuID => AtomsFineHalves[gpuID][h].FreeOnDevice(), MaxGPUs);
                    }

                    ParameterAccuracy = AccuracyValues.Select(v => MathHelper.Mean(v)).ToArray();

                    Console.Write("\n");
                }

                #endregion

                #region Optimization

                if (true)
                {
                    Console.WriteLine("  Optimizing:");

                    // For each half subset, take batches of particles and optimize the parameters for each body in them
                    for (int h = 0, completed = 0; h < 2; h++)
                    {
                        List<int> BatchStarts = new List<int>();
                        for (int batchStart = 0; batchStart < SubsetRows[h].Count; batchStart += BatchSize)
                            BatchStarts.Add(batchStart);

                        //for (int batchStart = 0; batchStart < SubsetRows[h].Count; batchStart += BatchSize)
                        Helper.ForEachGPU(BatchStarts, (batchStart, gpuID) =>
                        {
                            int CurBatch = Math.Min(BatchSize, SubsetRows[h].Count - batchStart);

                            Image CTFCoordsCoarse = CTF.GetCTFCoords(SizeCoarse, Size);
                            Image ParticleMask = ImageHelper.MakeCircularMask(SizeCoarse, ParticleDiameter / 2 / PixelSize * ScaleFactor, 10f);

                            #region Read particles from disk and scale them to SizeCoarse

                            Image ParticlesCoarseFT;
                            {
                                Image ParticlesRaw = new Image(new int3(Size, Size, CurBatch));
                                float[][] ParticlesRawData = ParticlesRaw.GetHost(Intent.Write);

                                for (int p = 0; p < CurBatch; p++)
                                {
                                    string ImageName = TableData.GetRowValue(SubsetRows[h][batchStart + p], "rlnImageName");

                                    if (!PrereadParticles)
                                    {
                                        string[] AddressParts = ImageName.Split(new[] { "@" }, StringSplitOptions.RemoveEmptyEntries);
                                        int Layer = int.Parse(AddressParts[0]) - 1;
                                        string FileName = ProjDir + AddressParts[1];

                                        Image Particle = Image.FromFile(FileName, new int2(Size, Size), 0, typeof (float), Layer);
                                        ParticlesRawData[p] = Particle.GetHostContinuousCopy();
                                        Particle.Dispose();
                                    }
                                    else
                                    {
                                        ParticlesRawData[p] = ParticleStorage[ImageName].ToArray();
                                    }

                                    // Divide by particle's norm correction
                                    float NormCorrection = 1f / ParticleNormalization[SubsetRows[h][batchStart + p]];
                                    for (int i = 0; i < ParticlesRawData[p].Length; i++)
                                        ParticlesRawData[p][i] *= NormCorrection;

                                }

                                Image ParticlesCoarse = ParticlesRaw.AsScaled(new int2(SizeCoarse, SizeCoarse));
                                ParticlesCoarse.Multiply(1f / (SizeCoarse * SizeCoarse));
                                //ParticlesCoarse.WriteMRC("d_particlescoarse.mrc");

                                ParticlesCoarse.MultiplySlices(ParticleMask);
                                ParticlesCoarse.RemapToFT();

                                ParticlesCoarseFT = ParticlesCoarse.AsFFT();
                                ParticlesCoarse.Dispose();

                                ParticlesRaw.Dispose();
                            }

                            #endregion

                            #region Generate per-particle 2D CTF

                            Image ParticlesCTF;
                            {
                                ParticlesCTF = new Image(ParticlesCoarseFT.Dims, true);
                                GPU.CreateCTF(ParticlesCTF.GetDevice(Intent.Write),
                                              CTFCoordsCoarse.GetDevice(Intent.Read),
                                              (uint)CTFCoordsCoarse.ElementsComplex,
                                              ParticleCTFParams[h].Skip(batchStart).Take(CurBatch).ToArray(),
                                              false,
                                              (uint)CurBatch);
                                //ParticlesCTF.WriteMRC("d_particlectf.mrc");

                                float[][] ParticlesCTFData = ParticlesCTF.GetHost(Intent.ReadWrite);
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float Scale = GroupScale[ParticleGroups[h][batchStart + p]];
                                    for (int i = 0; i < ParticlesCTFData[p].Length; i++)
                                        ParticlesCTFData[p][i] *= Scale;
                                }
                            }

                            #endregion

                            #region Generate per-particle 2D inverse noise spectrum
                            Image ParticlesInvSigma2;
                            {
                                float[][] InvSigma2Data = new float[CurBatch][];
                                for (int p = 0; p < CurBatch; p++)
                                    InvSigma2Data[p] = GroupInvNoise[ParticleGroups[h][batchStart + p]].ToArray();

                                ParticlesInvSigma2 = new Image(InvSigma2Data, new int3(SizeCoarse, SizeCoarse, CurBatch), true);
                            }
                            #endregion

                            int StepsPassed = 0;

                            double Delta = 0.025;
                            float Delta2 = 2 * (float)Delta;

                            #region Helper lambdas

                            Func<double[], bool, float[]> GetParticleDiffs = (input, doOnlyDiff2) =>
                            {
                                float2[] CurShifts;
                                float3[] CurAngles;
                                float[][] CurNMAWeights;
                                GetParametersFromVector(input, CurBatch, SizeCoarse / 2, NModes, out CurShifts, out CurAngles, out CurNMAWeights);

                                float[] PriorWeights = new float[CurBatch];
                                //Parallel.For(0, CurBatch, p =>
                                //{
                                //    float Displacement = AtomsCoarse.GetMeanDisplacement(CurNMAWeights[p]);
                                //    Displacement *= Displacement;

                                //    PriorWeights[p] = -Displacement / (2 * Sigma2NMA);
                                //});

                                Image ProjectionsFT = AtomsFineHalves[gpuID][h].ProjectBodies(new int2(SizeCoarse, SizeCoarse),
                                                                                              CurNMAWeights,
                                                                                              CurAngles.Select(v => v * Helper.ToRad).ToArray(),
                                                                                              CurShifts.Select(v => -v).ToArray(),
                                                                                              Helper.ArrayOfConstant(Helper.ArrayOfConstant(1f, AtomsFineHalves[gpuID][h].NSegments), CurBatch),
                                                                                              CurBatch);

                                //Projections.WriteMRC("d_projections.mrc");

                                float[] Diff2 = new float[CurBatch];

                                GPU.ParticleNMAGetDiff(ParticlesCoarseFT.GetDevice(Intent.Read),
                                                       ProjectionsFT.GetDevice(Intent.Read),
                                                       ParticlesCTF.GetDevice(Intent.Read),
                                                       ParticlesInvSigma2.GetDevice(Intent.Read),
                                                       new int2(SizeCoarse, SizeCoarse),
                                                       Diff2,
                                                       (uint)CurBatch);
                                ProjectionsFT.Dispose();

                                float[] Result = doOnlyDiff2 ? Diff2 : new float[CurBatch];

                                if (!doOnlyDiff2)
                                    for (int i = 0; i < CurBatch; i++)
                                        Result[i] = Diff2[i] - PriorWeights[i];

                                return Result;
                            };

                            #endregion

                            #region Gradient and eval lambdas

                            Func<double[], double> Eval = input =>
                            {
                                double Score = GetParticleDiffs(input, false).Sum();

                                if (double.IsNaN(Score))
                                    Debug.WriteLine($"{gpuID}: NaN!");

                                Debug.WriteLine($"{gpuID}: {Score / CurBatch}");

                                return Score;
                            };

                            Func<double[], double[]> Gradient = input =>
                            {
                                double[] Result = new double[input.Length];

                                if (StepsPassed++ > NSteps)
                                    return Result;

                                int PE = 5 + NModes;

                                for (int element = 0; element < PE; element++)
                                {
                                    double[] InputPlus = input.ToList().ToArray();
                                    for (int p = 0; p < CurBatch; p++)
                                        InputPlus[p * PE + element] += Delta;
                                    float[] ResultPlus = GetParticleDiffs(InputPlus, false);

                                    double[] InputMinus = input.ToList().ToArray();
                                    for (int p = 0; p < CurBatch; p++)
                                        InputMinus[p * PE + element] -= Delta;
                                    float[] ResultMinus = GetParticleDiffs(InputMinus, false);

                                    for (int p = 0; p < CurBatch; p++)
                                        Result[p * PE + element] = (ResultPlus[p] - ResultMinus[p]) / Delta2;
                                }

                                return Result;
                            };

                            #endregion

                            #region BFGS

                            double[] StartParams = GetVectorFromParameters(CurBatch,
                                                                           SizeCoarse / 2,
                                                                           NModes,
                                                                           ParticleShifts[h].Skip(batchStart).Take(CurBatch).ToArray(),
                                                                           ParticleAngles[h].Skip(batchStart).Take(CurBatch).ToArray(),
                                                                           ParticleNMAWeights[h].Skip(batchStart * NModes).Take(CurBatch * NModes).ToArray());
                            

                            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(CurBatch * (NModes + 5), Eval, Gradient);
                            Optimizer.Epsilon = 3e-7;
                            
                            Optimizer.Minimize(StartParams);

                            #endregion

                            #region Monte-Carlo sampling

                            {
                                RandomNormal Rand = new RandomNormal(iter * 999999 + batchStart);

                                float2[] OptimalShifts;
                                float3[] OptimalAngles;
                                float[][] OptimalNMAWeights;
                                GetParametersFromVector(StartParams, CurBatch, SizeCoarse / 2, NModes, out OptimalShifts, out OptimalAngles, out OptimalNMAWeights);

                                float[][] NormScores = Helper.ArrayOfFunction(() => new float[NSamplesMC], CurBatch);

                                for (int p = 0; p < CurBatch; p++)
                                {
                                    MCShifts[h][batchStart + p] = new float2[NSamplesMC];
                                    MCAngles[h][batchStart + p] = new float3[NSamplesMC];
                                    for (int m = 0; m < NModes; m++)
                                        MCNMAWeights[h][batchStart + p] = new float[NSamplesMC * NModes];
                                }

                                float AccuracyFraction = 3f;

                                for (int mcs = 0; mcs < NSamplesMC; mcs++)
                                {
                                    float2[] PerturbedShifts = new float2[CurBatch];
                                    float3[] PerturbedAngles = new float3[CurBatch];
                                    float[] PerturbedNMAWeights = new float[CurBatch * NModes];


                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        // Perturb currently optimal parameters using the previously 
                                        // estimated accuracy as the std dev of a normal distribution.
                                        // For the first sample, take the unperturbed optimal parameters.

                                        PerturbedShifts[p].X = Rand.NextSingle(OptimalShifts[p].X, mcs == 0 ? 0 : ParameterAccuracy[0] / AccuracyFraction);
                                        PerturbedShifts[p].Y = Rand.NextSingle(OptimalShifts[p].Y, mcs == 0 ? 0 : ParameterAccuracy[1] / AccuracyFraction);

                                        PerturbedAngles[p].X = Rand.NextSingle(OptimalAngles[p].X, mcs == 0 ? 0 : ParameterAccuracy[2] / AccuracyFraction);
                                        PerturbedAngles[p].Y = Rand.NextSingle(OptimalAngles[p].Y, mcs == 0 ? 0 : ParameterAccuracy[3] / AccuracyFraction);
                                        PerturbedAngles[p].Z = Rand.NextSingle(OptimalAngles[p].Z, mcs == 0 ? 0 : ParameterAccuracy[4] / AccuracyFraction);

                                        for (int m = 0; m < NModes; m++)
                                            PerturbedNMAWeights[p * NModes + m] = Rand.NextSingle(OptimalNMAWeights[p][m], mcs == 0 ? 0 : ParameterAccuracy[5 + m] / AccuracyFraction);
                                    }

                                    double[] PerturbedParams = GetVectorFromParameters(CurBatch,
                                                                                        SizeCoarse / 2,
                                                                                        NModes,
                                                                                        PerturbedShifts,
                                                                                        PerturbedAngles,
                                                                                        PerturbedNMAWeights);

                                    float[] PerturbedScores = GetParticleDiffs(PerturbedParams, false);

                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        MCShifts[h][batchStart + p][mcs] = PerturbedShifts[p];
                                        MCAngles[h][batchStart + p][mcs] = PerturbedAngles[p];
                                        for (int m = 0; m < NModes; m++)
                                            MCNMAWeights[h][batchStart + p][mcs * NModes + m] = PerturbedNMAWeights[p * NModes + m];

                                        NormScores[p][mcs] = PerturbedScores[p];
                                    }
                                }

                                float[] MinScore = NormScores.Select(a => MathHelper.Min(a)).ToArray();
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    for (int mcs = 0; mcs < NSamplesMC; mcs++)
                                    {
                                        NormScores[p][mcs] -= MinScore[p];
                                        NormScores[p][mcs] = NormScores[p][mcs] < 30 ? (float)Math.Exp(-NormScores[p][mcs]) : 0;
                                    }

                                    float ScoreSum = NormScores[p].Sum();
                                    for (int mcs = 0; mcs < NSamplesMC; mcs++)
                                        NormScores[p][mcs] /= ScoreSum;

                                    int[] BestIndices;
                                    float[] BestScores = MathHelper.TakeNHighest(NormScores[p], NSamplesMCSignificant, out BestIndices);

                                    MCScores[h][batchStart + p] = BestScores;
                                    MCShifts[h][batchStart + p] = Helper.IndexedSubset(MCShifts[h][batchStart + p], BestIndices);
                                    MCAngles[h][batchStart + p] = Helper.IndexedSubset(MCAngles[h][batchStart + p], BestIndices);

                                    int[] BestNMAIndices = Helper.Combine(BestIndices.Select(v => Helper.ArrayOfSequence(v * NModes, v * NModes + NModes, 1)));
                                    MCNMAWeights[h][batchStart + p] = Helper.IndexedSubset(MCNMAWeights[h][batchStart + p], BestNMAIndices);
                                }
                            }

                            #endregion

                            #region Set optimized parameters

                            //if (false)
                            {
                                float2[] CurShifts;
                                float3[] CurAngles;
                                float[][] CurNMAWeights;
                                GetParametersFromVector(StartParams, CurBatch, SizeCoarse / 2, NModes, out CurShifts, out CurAngles, out CurNMAWeights);

                                if (MathHelper.Max(Helper.Combine(CurNMAWeights)) < 0.1f)
                                    Debug.WriteLine("Something is wrong!");

                                lock (TableData)
                                {
                                    for (int m = 0; m < NModes; m++)
                                        if (!TableData.HasColumn($"wrpNMAWeight{m + 1}"))
                                            TableData.AddColumn($"wrpNMAWeight{m + 1}");
                                }

                                for (int r = 0; r < CurBatch; r++)
                                {
                                    int R = SubsetRows[h][batchStart + r];

                                    TableData.SetRowValue(R, "rlnOriginX", CurShifts[r].X);
                                    TableData.SetRowValue(R, "rlnOriginY", CurShifts[r].Y);

                                    TableData.SetRowValue(R, "rlnAngleRot", CurAngles[r].X);
                                    TableData.SetRowValue(R, "rlnAngleTilt", CurAngles[r].Y);
                                    TableData.SetRowValue(R, "rlnAnglePsi", CurAngles[r].Z);

                                    for (int m = 0; m < NModes; m++)
                                        TableData.SetRowValue(R, $"wrpNMAWeight{m + 1}", CurNMAWeights[r][m]);

                                    ParticleShifts[h][r + batchStart] = CurShifts[r];
                                    ParticleAngles[h][r + batchStart] = CurAngles[r];
                                    for (int m = 0; m < NModes; m++)
                                        ParticleNMAWeights[h][(r + batchStart) * NModes + m] = CurNMAWeights[r][m];
                                }
                            }

                            #endregion

                            ParticlesInvSigma2.Dispose();
                            ParticlesCoarseFT.Dispose();
                            ParticlesCTF.Dispose();
                            CTFCoordsCoarse.Dispose();
                            ParticleMask.Dispose();

                            lock (TableData)
                            {
                                completed += CurBatch;
                                ClearCurrentConsoleLine();
                                Console.Write($"    {completed}/{SubsetRows.Select(v => v.Count).Sum()}, {GPU.GetFreeMemory(0)} MB");
                                //Console.WriteLine(GPU.GetFreeMemory(0));
                            }
                        }, 1, MaxGPUs);

                        Helper.ForEachGPUOnce(gpuID => AtomsFineHalves[gpuID][h].FreeOnDevice(), MaxGPUs);

                        TableData.Save("d_tabledata.star");
                    }
                    Console.Write("\n");
                }

                #endregion

                Helper.ForEachGPUOnce(gpuID =>
                {
                    foreach (var gpuHalf in AtomsFineHalves[gpuID])
                    {
                        gpuHalf?.FreeOnDevice();
                        gpuHalf?.Dispose();
                    }
                }, MaxGPUs);

                #region Reconstruct half-maps for each body considering normal modes, combine in Frankenmaps

                Console.WriteLine("  Making reconstructions:");

                // Normalize group scales to have average of 1.0
                {
                    float AverageScale = MathHelper.Mean(GroupScale);
                    GroupScale = GroupScale.Select(v => v / AverageScale).ToArray();
                }

                Image[] FrankenParts1 = null, FrankenParts2 = null;
                Image[] FrankenMasks1 = null, FrankenMasks2 = null;
                Image FrankenMap1 = null, FrankenMap2 = null;

                {
                    int completed = 0;
                    Helper.ForGPU(0, 2, (h, gpuID) =>
                    {
                        NMAMap AtomsFrankenMap = AtomsCoarse.GetCopy();
                        if (ForcedSegmentation == null)
                            AtomsFrankenMap.ReduceToNSegments(NSegmentsReconstruction);
                        else
                            AtomsFrankenMap.AddSegmentation(new Image(ForcedSegmentation, new int3(Size, Size, Size)));

                        AtomsFrankenMap.InitializeBodyReconstructions(Helper.ArrayOfSequence(0, AtomsFrankenMap.NSegments, 1), 2);

                        List<int> BatchStarts = new List<int>();
                        for (int batchStart = 0; batchStart < SubsetRows[h].Count; batchStart += BatchSize)
                            BatchStarts.Add(batchStart);

                        //Image[] ParticleFTStorage = new Image[BatchStarts.Count];
                        //Image[] ParticleCTFStorage = new Image[BatchStarts.Count];

                        int SimultaneousRecs = 4;
                        for (int irec = 0; irec < AtomsFrankenMap.NSegments; irec += SimultaneousRecs)
                        {
                            int CurNumRecs = Math.Min(SimultaneousRecs, AtomsFrankenMap.NSegments - irec);
                            int[] RecIndices = Helper.ArrayOfSequence(irec, irec + CurNumRecs, 1);

                            //Parallel.ForEach(BatchStarts, new ParallelOptions { MaxDegreeOfParallelism = 4 }, batchStart =>
                            foreach (int batchStart in BatchStarts)
                            {
                                int CurBatch = Math.Min(BatchSize, SubsetRows[h].Count - batchStart);

                                #region Read particles

                                Image ParticlesFT;
                                //if (!PrereadParticles || irec == 0)
                                {
                                    Image Particles = new Image(new int3(Size, Size, CurBatch));
                                    float[][] ParticlesData = Particles.GetHost(Intent.Write);

                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        string ImageName = TableData.GetRowValue(SubsetRows[h][batchStart + p], "rlnImageName");

                                        if (!PrereadParticles)
                                        {
                                            string[] AddressParts = ImageName.Split(new[] { "@" }, StringSplitOptions.RemoveEmptyEntries);
                                            int Layer = int.Parse(AddressParts[0]) - 1;
                                            string FileName = ProjDir + AddressParts[1];

                                            Image Particle = Image.FromFile(FileName, new int2(Size, Size), 0, typeof (float), Layer);
                                            ParticlesData[p] = Particle.GetHostContinuousCopy();
                                            Particle.Dispose();
                                        }
                                        else
                                        {
                                            ParticlesData[p] = ParticleStorage[ImageName].ToArray();
                                        }

                                        // Divide by particle's norm correction
                                        float NormCorrection = 1f / ParticleNormalization[SubsetRows[h][batchStart + p]] / GroupScale[ParticleGroups[h][batchStart + p]];
                                        for (int i = 0; i < ParticlesData[p].Length; i++)
                                            ParticlesData[p][i] *= NormCorrection;
                                    }

                                    Particles.RemapToFT();
                                    ParticlesFT = Particles.AsFFT();
                                    Particles.Dispose();

                                    //if (PrereadParticles)
                                    //    ParticleFTStorage[batchStart / BatchSize] = ParticlesFT;
                                }
                                //else
                                //{
                                //    ParticlesFT = ParticleFTStorage[batchStart / BatchSize];
                                //}

                                #endregion

                                #region Generate per-particle 2D CTF

                                Image ParticlesCTFSq;
                                //if (!PrereadParticles || irec == 0)
                                {
                                    Image CTFCoords = CTF.GetCTFCoords(Size, Size);
                                    Image ParticlesCTF = new Image(ParticlesFT.Dims, true);
                                    GPU.CreateCTF(ParticlesCTF.GetDevice(Intent.Write),
                                                  CTFCoords.GetDevice(Intent.Read),
                                                  (uint)ParticlesCTF.ElementsSliceReal,
                                                  ParticleCTFParams[h].Skip(batchStart).Take(CurBatch).ToArray(),
                                                  false,
                                                  (uint)CurBatch);
                                    //ParticlesCTF.WriteMRC("d_particlectf.mrc");

                                    ParticlesFT.Multiply(ParticlesCTF);

                                    ParticlesCTFSq = new Image(ParticlesCTF.GetDevice(Intent.Read), ParticlesCTF.Dims, true);
                                    ParticlesCTFSq.Multiply(ParticlesCTFSq);

                                    ParticlesCTF.Dispose();
                                    CTFCoords.Dispose();

                                    //if (PrereadParticles)
                                    //    ParticleCTFStorage[batchStart / BatchSize] = ParticlesCTFSq;
                                }
                                //else
                                //{
                                //    ParticlesCTFSq = ParticleCTFStorage[batchStart / BatchSize];
                                //}

                                #endregion

                                #region Back-project for each MC sample

                                for (int mcs = 0; mcs < NSamplesMCSignificant; mcs++)
                                {
                                    float3[] Angles = new float3[CurBatch];
                                    float2[] Shifts = new float2[CurBatch];
                                    float[][] NMAWeights = Helper.ArrayOfFunction(() => new float[NModes], CurBatch);
                                    float[][] BodyWeights = Helper.ArrayOfFunction(() => new float[CurNumRecs], CurBatch);

                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        Angles[p] = MCAngles[h][batchStart + p][mcs] * Helper.ToRad;
                                        Shifts[p] = MCShifts[h][batchStart + p][mcs];

                                        BodyWeights[p] = Helper.ArrayOfConstant(MCScores[h][batchStart + p][mcs], CurNumRecs);

                                        for (int m = 0; m < NModes; m++)
                                            NMAWeights[p][m] = MCNMAWeights[h][batchStart + p][mcs * NModes + m];
                                    }

                                    AtomsFrankenMap.BackProjectBodies(RecIndices, ParticlesFT, ParticlesCTFSq, NMAWeights, Angles, Shifts, BodyWeights, CurBatch);
                                }

                                #endregion
                                
                                ParticlesCTFSq.Dispose();
                                ParticlesFT.Dispose();

                                lock (TableData)
                                {
                                    completed += CurBatch * CurNumRecs;
                                    ClearCurrentConsoleLine();
                                    Console.Write($"    {completed}/{SubsetRows.Select(v => v.Count).Sum() * AtomsFrankenMap.NSegments}, {GPU.GetFreeMemory(0)} MB");
                                }
                            }//);

                            foreach (var recIndex in RecIndices)
                                AtomsFrankenMap.BodyReconstructions[recIndex].FreeDevice();
                        }

                        Image[] FrankenParts = AtomsFrankenMap.ReconstructAllBodies(); //BodyOverlap, BodyOverlap * 3f
                        Image[] FrankenMasks = AtomsFrankenMap.GetSoftSegmentMasks(BodyOverlap, BodyOverlap * 2f, true);
                        AtomsFrankenMap.FreeOnDevice();
                        AtomsFrankenMap.Dispose();

                        Image FrankenMap = new Image(FrankenParts[0].Dims);
                        for (int i = 0; i < FrankenParts.Length; i++)
                        {
                            Image PartCopy = new Image(FrankenParts[i].GetDevice(Intent.Read), FrankenParts[i].Dims);
                            PartCopy.Multiply(FrankenMasks[i]);
                            FrankenMap.Add(PartCopy);

                            FrankenParts[i].FreeDevice();
                            FrankenMasks[i].FreeDevice();
                            PartCopy.Dispose();
                        }
                        FrankenMap.FreeDevice();

                        //FrankenMap.WriteMRC($"d_frankenmap_half{h}.mrc");

                        if (h == 0)
                        {
                            FrankenParts1 = FrankenParts;
                            FrankenMasks1 = FrankenMasks;
                            FrankenMap1 = FrankenMap;
                        }
                        else
                        {
                            FrankenParts2 = FrankenParts;
                            FrankenMasks2 = FrankenMasks;
                            FrankenMap2 = FrankenMap;
                        }
                    }, 1, MaxGPUs);
                }

                Console.Write("\n");
                #endregion

                #region Postprocess Frankenmaps, estimate local resolution

                {
                    // Average half-maps up to 40 A to prevent them from diverging
                    //Image FrankenMap1Averaged, FrankenMap2Averaged;
                    //FSC.AverageLowFrequencies(FrankenMap1, FrankenMap2, (int)(Size * PixelSize / 40), out FrankenMap1Averaged, out FrankenMap2Averaged);
                    //FrankenMap1.Dispose();
                    //FrankenMap1 = FrankenMap1Averaged;
                    //FrankenMap2.Dispose();
                    //FrankenMap2 = FrankenMap2Averaged;

                    FrankenMap1.WriteMRC(ProjDir + $"it{iter + 1:D3}_frankenmap_pristine_half1.mrc");
                    FrankenMap2.WriteMRC(ProjDir + $"it{iter + 1:D3}_frankenmap_pristine_half2.mrc");

                    if (!TableMap.HasColumn("wrpMapHalfPristine1"))
                        TableMap.AddColumn("wrpMapHalfPristine1");
                    if (!TableMap.HasColumn("wrpMapHalfPristine2"))
                        TableMap.AddColumn("wrpMapHalfPristine2");
                    TableMap.SetRowValue(0, "wrpMapHalfPristine1", $"it{iter + 1:D3}_frankenmap_pristine_half1.mrc");
                    TableMap.SetRowValue(0, "wrpMapHalfPristine2", $"it{iter + 1:D3}_frankenmap_pristine_half2.mrc");

                    Image LocalFilt = new Image(IntPtr.Zero, new int3(Size, Size, Size));
                    Image LocalRes = new Image(IntPtr.Zero, new int3(Size, Size, Size));

                    // Calculate resolution for each of the superfine bodies
                    // and create the localfilt map (local res, sharpened)
                    {
                        Console.WriteLine("  Estimating final resolution:");

                        GPU.LocalRes(FrankenMap1.GetDevice(Intent.Read),
                                     FrankenMap2.GetDevice(Intent.Read),
                                     new int3(Size, Size, Size),
                                     PixelSize,
                                     IntPtr.Zero,
                                     LocalFilt.GetDevice(Intent.Write),
                                     LocalRes.GetDevice(Intent.Write),
                                     IntPtr.Zero,
                                     FilterWindowSize,
                                     FSCThreshold,
                                     false,
                                     10,
                                     OutputBFactor,
                                     0,
                                     false,
                                     false);

                        float[] LocalResData = LocalRes.GetHostContinuousCopy().Select(v => Size * PixelSize / Math.Max(1, v)).ToArray();   // as frequency values now

                        LocalFilt.WriteMRC(ProjDir + $"it{iter + 1:D3}_frankenmap_localfilt.mrc");
                        LocalRes.WriteMRC(ProjDir + $"it{iter + 1:D3}_frankenmap_localres.mrc");

                        TableMap.SetRowValue(0, "wrpFrankenmap", $"it{iter + 1:D3}_frankenmap_localfilt.mrc");

                        Image[] MasksHard = AtomsCoarse.GetSegmentMasks();
                        int Done = 0;
                        Parallel.For(0, AtomsCoarse.NSegments, b =>
                        {
                            Image MaskHard = MasksHard[b];
                            float[] MaskHardData = MaskHard.GetHostContinuousCopy();

                            float AverageShell = MathHelper.MeanWeighted(LocalResData, MaskHardData);

                            TableBodies.SetRowValue(b, "wrpBodyResolution", $"{Size * PixelSize / Math.Max(1, AverageShell):F2}"); // back to wavelength

                            MaskHard.Dispose();

                            lock (TableData)
                            {
                                ClearCurrentConsoleLine();
                                Console.Write($"    {++Done}/{AtomsCoarse.NSegments}");
                            }
                        });

                        Console.Write("\n");
                    }

                    // Calculate resolution for each of the reconstruction bodies
                    // and average low frequencies of each body to keep things in place
                    {
                        Console.WriteLine("  Averaging low frequencies to prevent drifting:");

                        GPU.LocalRes(FrankenMap1.GetDevice(Intent.Read),
                                     FrankenMap2.GetDevice(Intent.Read),
                                     new int3(Size, Size, Size),
                                     PixelSize,
                                     IntPtr.Zero,
                                     LocalFilt.GetDevice(Intent.Write),
                                     LocalRes.GetDevice(Intent.Write),
                                     IntPtr.Zero,
                                     FilterWindowSize,
                                     0.95f,
                                     false,
                                     10,
                                     0,
                                     0,
                                     false,
                                     false);

                        float[] LocalResData = LocalRes.GetHostContinuousCopy().Select(v => Size * PixelSize / Math.Max(1, v)).ToArray();   // as frequency values now

                        NMAMap AtomsReduced = AtomsCoarse.GetCopy();
                        AtomsReduced.ReduceToNSegments(NSegmentsReconstruction);

                        float[] AverageShells = new float[NSegmentsReconstruction];

                        Image[] MasksHard = AtomsReduced.GetSegmentMasks();
                        AtomsReduced.Dispose();
                        int Done = 0;
                        Parallel.For(0, NSegmentsReconstruction, b =>
                        {
                            Image MaskHard = MasksHard[b];
                            float[] MaskHardData = MaskHard.GetHostContinuousCopy();

                            AverageShells[b] = Math.Min(Size * PixelSize / 25, MathHelper.MeanWeighted(LocalResData, MaskHardData));

                            MaskHard.Dispose();

                            lock (TableData)
                            {
                                ClearCurrentConsoleLine();
                                Console.Write($"    {++Done}/{NSegmentsReconstruction * 2}");
                            }
                        });

                        FrankenMap1.Dispose();
                        FrankenMap1 = new Image(FrankenParts1[0].Dims);
                        FrankenMap2.Dispose();
                        FrankenMap2 = new Image(FrankenParts2[0].Dims);

                        for (int b = 0; b < NSegmentsReconstruction; b++)
                        {
                            Image Part1Averaged, Part2Averaged;
                            FSC.AverageLowFrequencies(FrankenParts1[b], FrankenParts2[b], (int)AverageShells[b], out Part1Averaged, out Part2Averaged);

                            Part1Averaged.Multiply(FrankenMasks1[b]);
                            FrankenMap1.Add(Part1Averaged);
                            FrankenParts1[b].Dispose();
                            FrankenMasks1[b].Dispose();
                            Part1Averaged.Dispose();

                            Part2Averaged.Multiply(FrankenMasks2[b]);
                            FrankenMap2.Add(Part2Averaged);
                            FrankenParts2[b].Dispose();
                            FrankenMasks2[b].Dispose();
                            Part2Averaged.Dispose();

                            lock (TableData)
                            {
                                ClearCurrentConsoleLine();
                                Console.Write($"    {++Done}/{NSegmentsReconstruction * 2}");
                            }
                        }

                        Console.Write("\n");
                    }

                    LocalFilt.Dispose();
                    LocalRes.Dispose();

                    FrankenMap1.WriteMRC(ProjDir + $"it{iter + 1:D3}_frankenmap_half1.mrc");
                    FrankenMap2.WriteMRC(ProjDir + $"it{iter + 1:D3}_frankenmap_half2.mrc");

                    FrankenMap1.Dispose();
                    FrankenMap2.Dispose();

                    TableMap.SetRowValue(0, "wrpMapHalf1", $"it{iter + 1:D3}_frankenmap_half1.mrc");
                    TableMap.SetRowValue(0, "wrpMapHalf2", $"it{iter + 1:D3}_frankenmap_half2.mrc");
                }

                FrankenMap1.Dispose();
                FrankenMap2.Dispose();

                #endregion

                #region Write out tables

                TableData.Save(ProjDir + $"it{iter + 1:D3}_data.star");
                TableMap.SetRowValue(0, "wrpParticles", $"it{iter + 1:D3}_data.star");

                Star.SaveMultitable(ProjDir + $"it{iter + 1:D3}_model.star", new Dictionary<string, Star> { {"map", TableMap}, { "atoms", TableAtoms }, { "bodies", TableBodies } });
                ModelStarPath = ProjDir + $"it{iter + 1:D3}_model.star";

                #endregion

                // Report resolution for each body
                Console.WriteLine($"Finished iteration {iter + 1}, resolution for individual bodies at {FSCThreshold:F3} cutoff is:");
                for (int b = 0; b < NBodies; b++)
                    Console.WriteLine($"  {b + 1}: {TableBodies.GetRowValue(b, "wrpBodyResolution")} A");

                Console.WriteLine("");
                Console.WriteLine("");

                NSegmentsOptimization = Math.Min(NMaxSegments, NSegmentsOptimization + 2);
                NSegmentsReconstruction = Math.Min(NMaxSegments, NSegmentsReconstruction + 2);
            }
        }

        private static double[] GetVectorFromParameters(int nparticles, int sizeParticle, int nmodes, float2[] particleShifts, float3[] particleAngles, float[] particleNMAWeights)
        {
            float AverageMoveParticle = sizeParticle / 180f;
            int PE = 5 + nmodes;
            double[] Result = new double[nparticles * PE];

            for (int p = 0; p < nparticles; p++)
            {
                Result[p * PE + 0] = particleShifts[p].X;
                Result[p * PE + 1] = particleShifts[p].Y;

                Result[p * PE + 2] = particleAngles[p].X * AverageMoveParticle;
                Result[p * PE + 3] = particleAngles[p].Y * AverageMoveParticle;
                Result[p * PE + 4] = particleAngles[p].Z * AverageMoveParticle;

                for (int m = 0; m < nmodes; m++)
                    Result[p * PE + 5 + m] = particleNMAWeights[p * nmodes + m];
            }

            return Result;
        }

        private static void GetParametersFromVector(double[] v, int nparticles, int sizeParticle, int nmodes, out float2[] particleShifts, out float3[] particleAngles, out float[][] particleNMAWeights)
        {
            float AverageMoveParticle = sizeParticle / 180f;
            int PE = 5 + nmodes;

            particleShifts = new float2[nparticles];
            particleAngles = new float3[nparticles];
            particleNMAWeights = Helper.ArrayOfFunction(() => new float[nmodes], nparticles);
            
            for (int p = 0; p < nparticles; p++)
            {
                particleShifts[p] = new float2((float)v[p * PE + 0], (float)v[p * PE + 1]);
                particleAngles[p] = new float3((float)v[p * PE + 2], (float)v[p * PE + 3], (float)v[p * PE + 4]) / AverageMoveParticle;

                for (int m = 0; m < nmodes; m++)
                    particleNMAWeights[p][m] = (float)v[p * PE + 5 + m];
            }
        }

        private static void ClearCurrentConsoleLine()
        {
            int currentLineCursor = Console.CursorTop;
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write(new string(' ', Console.WindowWidth));
            Console.SetCursorPosition(0, currentLineCursor);
        }
    }
}
