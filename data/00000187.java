public class Font { private byte[] bs; private TableDirectory tableDirectory = null; private Table[] tables; private Os2Table os2; private CmapTable cmap; private GlyfTable glyf; private HeadTable head; private HheaTable hhea; private HmtxTable hmtx; private LocaTable loca; private MaxpTable maxp; private NameTable name; private PostTable post; public Font() { } public Table getTable(int tableType) { for (int i = 0; i < tables.length; i++) { if ((tables[i] != null) && (tables[i].getType() == tableType)) { return tables[i]; } } return null; } public Os2Table getOS2Table() { return os2; } public CmapTable getCmapTable() { return cmap; } public HeadTable getHeadTable() { return head; } public HheaTable getHheaTable() { return hhea; } public HmtxTable getHmtxTable() { return hmtx; } public LocaTable getLocaTable() { return loca; } public MaxpTable getMaxpTable() { return maxp; } public NameTable getNameTable() { return name; } public PostTable getPostTable() { return post; } public int getAscent() { return hhea.getAscender(); } public int getDescent() { return hhea.getDescender(); } public int getNumGlyphs() { return maxp.getNumGlyphs(); } public Glyph getGlyph(int i) { return (glyf.getDescription(i) != null) ? new Glyph( glyf.getDescription(i), hmtx.getLeftSideBearing(i), hmtx.getAdvanceWidth(i)) : null; } public TableDirectory getTableDirectory() { return tableDirectory; } protected void read(byte[] fontInBytes) { bs = fontInBytes; try { RandomAccessFileEmulator raf = new RandomAccessFileEmulator(bs, "r"); tableDirectory = new TableDirectory(raf); tables = new Table[tableDirectory.getNumTables()]; for (int i = 0; i < tableDirectory.getNumTables(); i++) { tables[i] = TableFactory.create (tableDirectory.getEntry(i), raf); } raf.close(); os2 = (Os2Table) getTable(Table.OS_2); cmap = (CmapTable) getTable(Table.cmap); glyf = (GlyfTable) getTable(Table.glyf); head = (HeadTable) getTable(Table.head); hhea = (HheaTable) getTable(Table.hhea); hmtx = (HmtxTable) getTable(Table.hmtx); loca = (LocaTable) getTable(Table.loca); maxp = (MaxpTable) getTable(Table.maxp); name = (NameTable) getTable(Table.name); post = (PostTable) getTable(Table.post); hmtx.init(hhea.getNumberOfHMetrics(), maxp.getNumGlyphs() - hhea.getNumberOfHMetrics()); loca.init(maxp.getNumGlyphs(), head.getIndexToLocFormat() == 0); glyf.init(maxp.getNumGlyphs(), loca); } catch (IOException e) { e.printStackTrace(); } } public static Font create() { return new Font(); } public static Font create(byte[] fontInBytes) { Font f = new Font(); f.read(fontInBytes); return f; } }