public class FpgmTable extends Program implements Table { protected FpgmTable ( DirectoryEntry de , RandomAccessFileEmulator raf ) throws IOException { raf . seek ( de . getOffset ( ) ) ; readInstructions ( raf , de . getLength ( ) ) ; } public int getType ( ) { return fpgm ; } }