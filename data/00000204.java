public class ScriptRecord { private int tag ; private int offset ; protected ScriptRecord ( RandomAccessFileEmulator raf ) throws IOException { tag = raf . readInt ( ) ; offset = raf . readUnsignedShort ( ) ; } public int getTag ( ) { return tag ; } public int getOffset ( ) { return offset ; } }