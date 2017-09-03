using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;

namespace Warp.Craft
{
    class Options
    {
        [Option("input", Required = true, HelpText = "Relative path to a STAR file with initial model parameters.")]
        public string ModelStarPath { get; set; }

        [Option("iterations", DefaultValue = 5, HelpText = "Number of iterations.")]
        public int NIterations { get; set; }

        [Option("steps", DefaultValue = 40, HelpText = "Number of optimization steps per iteration.")]
        public int NSteps { get; set; }

        [Option("batchsize", DefaultValue = 300, HelpText = "Number of particles to be processed simultaneously in one GPU thread.")]
        public int BatchSize { get; set; }

        [Option("particlediameter", DefaultValue = 200f, HelpText = "Particle diameter in Angstrom.")]
        public float ParticleDiameter { get; set; }

        [Option("angpix", DefaultValue = 2.74f, HelpText = "Pixel size in Angstrom.")]
        public float PixelSize { get; set; }

        [Option("gputhreads", DefaultValue = 1, HelpText = "Number of concurrent threads per GPU.")]
        public int ThreadsPerGPU { get; set; }

        [Option("bfactor", DefaultValue = 0, HelpText = "B-factor in A^2 for locally filtered output, more negative = sharper.")]
        public int BFactor { get; set; }

        [Option("filterwindow", DefaultValue = 40, HelpText = "Window size in pixels for the local resolution estimation and filtering.")]
        public int WindowSize { get; set; }

        [Option("fscthreshold", DefaultValue = 0.5f, HelpText = "FSC threshold for the local resolution estimation.")]
        public float FSCThreshold { get; set; }

        [Option("dilatemask", DefaultValue = 5, HelpText = "Dilate mask by this many voxels before constructing the fine atom model.")]
        public int DilateMask { get; set; }

        [Option("bodyoverlap", DefaultValue = 2, HelpText = "Region of overlap between bodies in voxels.")]
        public int BodyOverlap { get; set; }

        [Option("initcoarse", DefaultValue = "", HelpText = "Initialize normal modes with this model to save time instead of calculating it from scratch.")]
        public string InitCoarsePath { get; set; }

        [Option("ncoarseatoms", DefaultValue = 5000, HelpText = "Number of atoms used to compute the normal modes. Will be interpolated onto the fine model later.")]
        public int NCoarseAtoms { get; set; }

        [Option("oversamplefineatoms", DefaultValue = 4, HelpText = "Oversampling factor for the fine atom model, N x number of voxels in mask.")]
        public int OversampleFine { get; set; }

        [Option("updatemodes", DefaultValue = false, HelpText = "Update normal modes after each iteration. This can take a lot of time.")]
        public bool UpdateModes { get; set; }

        [Option("nmodes", DefaultValue = 10, HelpText = "Number of normal modes to use. Low-frequency modes come first.")]
        public int NModes { get; set; }

        [Option("prereadparticles", DefaultValue = true, HelpText = "Pre-read all particles into RAM to increase IO speed.")]
        public bool PrereadParticles { get; set; }

        [ParserState]
        public IParserState LastParserState { get; set; }

        [HelpOption]
        public string GetUsage()
        {
            return HelpText.AutoBuild(this, (HelpText current) => HelpText.DefaultParsingErrorsHandler(this, current));
        }
    }
}
