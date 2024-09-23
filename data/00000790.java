public class PwDbV3OutputDebug extends PwDbV3Output { PwDatabaseV3Debug debugDb; private boolean noHeaderHash; public PwDbV3OutputDebug(PwDatabaseV3 pm, OutputStream os) { this(pm, os, false); } public PwDbV3OutputDebug(PwDatabaseV3 pm, OutputStream os, boolean noHeaderHash) { super(pm, os); debugDb = (PwDatabaseV3Debug) pm; this.noHeaderHash = noHeaderHash; } @Override protected SecureRandom setIVs(PwDbHeader h) throws PwDbOutputException { PwDbHeaderV3 header = (PwDbHeaderV3) h; PwDbHeaderV3 origHeader = debugDb.dbHeader; System.arraycopy(origHeader.encryptionIV, 0, header.encryptionIV, 0, origHeader.encryptionIV.length); System.arraycopy(origHeader.masterSeed, 0, header.masterSeed, 0, origHeader.masterSeed.length); System.arraycopy(origHeader.transformSeed, 0, header.transformSeed, 0, origHeader.transformSeed.length); return null; } @Override protected boolean useHeaderHash() { return !noHeaderHash; } }