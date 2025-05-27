
import osprey
import multiprocessing
import time

# Start timer
start_time = time.time()

# Set CPU cores (leave 2 cores free)
cpu_cores = 10  # Matches 10 physical cores

# Initialize OSPREY with 20 GB heap
osprey.start(heapSizeMiB=20000)

# Enable external memory for large calculations
osprey.initExternalMemory(internalSizeMiB=12000) #12 GB external memory

# Setup forcefield and load PDB
ffparams = osprey.ForcefieldParams()
mol = osprey.readPdb('4wrm.renum.pdb')
templateLib = osprey.TemplateLibrary(ffparams.forcefld)

# Define protein strand (A141 to A152)
protein = osprey.Strand(mol, templateLib=templateLib, residues=['A230', 'A258'])
# Set flexibility and mutations
protein.flexibility['A231'].setLibraryRotamers(osprey.WILD_TYPE, 'LEU', 'ILE', 'ALA').addWildTypeRotamers().setContinuous() #Valine->Leu, Ile, Ala
protein.flexibility['A234'].setLibraryRotamers(osprey.WILD_TYPE, 'GLU', 'ASN', 'GLN', 'SER').addWildTypeRotamers().setContinuous() #Aspartic acid->Glu, Asn, Gln
protein.flexibility['A250'].setLibraryRotamers(osprey.WILD_TYPE, 'THR', 'ASN', 'CYS').addWildTypeRotamers().setContinuous() #Serine->Thr, Asn, Cys
protein.flexibility['A251'].setLibraryRotamers(osprey.WILD_TYPE, 'GLU', 'ASN', 'SER','GLN').addWildTypeRotamers().setContinuous() #Aspartic acid->Glu, Asn, Ser
protein.flexibility['A252'].setLibraryRotamers(osprey.WILD_TYPE, 'TYR', 'TRP', 'LEU').addWildTypeRotamers().setContinuous() #Phenylalanin->Tyr, Trp, Leu
protein.flexibility['A257'].setLibraryRotamers(osprey.WILD_TYPE, 'PHE', 'TRP', 'HIS').addWildTypeRotamers().setContinuous() #Tyrosine->Phe, Trp, His

# Define ligand strand (adjust based on PDB)
ligand = osprey.Strand(mol, templateLib=templateLib, residues=['B5', 'B16'])
ligand.flexibility['B6'].setLibraryRotamers(osprey.WILD_TYPE).addWildTypeRotamers().setContinuous()
ligand.flexibility['B9'].setLibraryRotamers(osprey.WILD_TYPE).addWildTypeRotamers().setContinuous()
ligand.flexibility['B10'].setLibraryRotamers(osprey.WILD_TYPE).addWildTypeRotamers().setContinuous()
ligand.flexibility['B13'].setLibraryRotamers(osprey.WILD_TYPE).addWildTypeRotamers().setContinuous()
ligand.flexibility['B14'].setLibraryRotamers(osprey.WILD_TYPE).addWildTypeRotamers().setContinuous()
ligand.flexibility['B15'].setLibraryRotamers(osprey.WILD_TYPE).addWildTypeRotamers().setContinuous()

# Create configuration spaces
proteinConfSpace = osprey.ConfSpace(protein)
ligandConfSpace = osprey.ConfSpace(ligand)
complexConfSpace = osprey.ConfSpace([protein, ligand])

# Setup energy calculator
parallelism = osprey.Parallelism(cpuCores=10)
ecalc = osprey.EnergyCalculator(complexConfSpace, ffparams, parallelism=parallelism)

# Configure K*
kstar = osprey.KStar(
    proteinConfSpace,
    ligandConfSpace,
    complexConfSpace,
    epsilon=0.1,  # Coarse for testing
    maxSimultaneousMutations=6,  # Limit to 2 mutations
    writeSequencesToConsole=True,
    writeSequencesToFile='csf1r_kstar_results.tsv'
)

# Setup partition functions
for info in kstar.confSpaceInfos():
    eref = osprey.ReferenceEnergies(info.confSpace, ecalc)
    info.confEcalc = osprey.ConfEnergyCalculator(info.confSpace, ecalc, referenceEnergies=eref)
    emat = osprey.EnergyMatrix(info.confEcalc, cacheFile='emat.%s.dat' % info.id)
    def makePfunc(rcs, confEcalc=info.confEcalc, emat=emat):
        return osprey.PartitionFunction(
            confEcalc,
            osprey.AStarTraditional(emat, rcs, showProgress=True),
            osprey.AStarTraditional(emat, rcs, showProgress=True),
            rcs
        )
    info.pfuncFactory = osprey.KStar.PfuncFactory(makePfunc)

# Run K*
scoredSequences = kstar.run(ecalc.tasks)

# Analyze results
analyzer = osprey.SequenceAnalyzer(kstar)
for scoredSequence in scoredSequences:
    print("result:")
    print("\tsequence: %s" % scoredSequence.sequence)
    print("\tK* score: %s" % scoredSequence.score)
    numConfs = 10
    analysis = analyzer.analyze(scoredSequence.sequence, numConfs)
    print(analysis)
    analysis.writePdb(
        'csf1r_seq.%s.pdb' % scoredSequence.sequence,
        'Top %d conformations for CSF1R sequence %s' % (numConfs, scoredSequence.sequence)
    )
