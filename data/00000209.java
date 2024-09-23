public class LigatureSet { private int ligatureCount; private int[] ligatureOffsets; private Ligature[] ligatures; public LigatureSet(RandomAccessFileEmulator raf, int offset) throws IOException { raf.seek(offset); ligatureCount = raf.readUnsignedShort(); ligatureOffsets = new int[ligatureCount]; ligatures = new Ligature[ligatureCount]; for (int i = 0; i < ligatureCount; i++) { ligatureOffsets[i] = raf.readUnsignedShort(); } for (int i = 0; i < ligatureCount; i++) { raf.seek(offset + ligatureOffsets[i]); ligatures[i] = new Ligature(raf); } } }