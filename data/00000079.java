public class RequestMessage extends BaseMessage{ private boolean hasBackfill; private boolean hasVBucketList; private boolean hasVBucketCheckpoints; private boolean hasFlags; private List<TapRequestFlag> flagList; private short[] vblist; private String name; private long backfilldate; private Map<Short, Long> vBucketCheckpoints; public RequestMessage() { flagList = new LinkedList<TapRequestFlag>(); vblist = new short[0]; vBucketCheckpoints = new HashMap<Short, Long>(); name = UUID.randomUUID().toString(); backfilldate = -1; totalbody += name.length(); keylength = (short) name.length(); } public void setFlags(TapRequestFlag f) { if (!flagList.contains(f)) { if (!hasFlags) { hasFlags = true; extralength += 4; totalbody += 4; } if (f.equals(TapRequestFlag.BACKFILL)) { hasBackfill = true; totalbody += 8; } if (f.equals(TapRequestFlag.LIST_VBUCKETS) || f.equals(TapRequestFlag.TAKEOVER_VBUCKETS)) { hasVBucketList = true; totalbody += 2; } if (f.equals(TapRequestFlag.CHECKPOINT)) { hasVBucketCheckpoints = true; totalbody += 2; } flagList.add(f); } } public List<TapRequestFlag> getFlags() { return flagList; } public void setBackfill(long date) { backfilldate = date; } public void setVbucketlist(short[] vbs) { int oldSize = (vblist.length + 1) * 2; int newSize = (vbs.length + 1) * 2; totalbody += newSize - oldSize; vblist = vbs; } public void setvBucketCheckpoints(Map<Short, Long> vbchkpnts) { int oldSize = (vBucketCheckpoints.size()) * 10; int newSize = (vbchkpnts.size()) * 10; totalbody += newSize - oldSize; vBucketCheckpoints = vbchkpnts; } public void setName(String n) { if (n.length() > 65535) { throw new IllegalArgumentException("Tap name too long"); } totalbody += n.length() - name.length(); keylength = (short) n.length(); name = n; } @Override public ByteBuffer getBytes() { ByteBuffer bb = ByteBuffer.allocate(HEADER_LENGTH + getTotalbody()); bb.put(magic.getMagic()); bb.put(opcode.getOpcode()); bb.putShort(keylength); bb.put(extralength); bb.put(datatype); bb.putShort(vbucket); bb.putInt(totalbody); bb.putInt(opaque); bb.putLong(cas); if (hasFlags) { int flag = 0; for (int i = 0; i < flagList.size(); i++) { flag |= flagList.get(i).getFlags(); } bb.putInt(flag); } bb.put(name.getBytes()); if (hasBackfill) { bb.putLong(backfilldate); } if (hasVBucketList) { bb.putShort((short) vblist.length); for (int i = 0; i < vblist.length; i++) { bb.putShort(vblist[i]); } } if (hasVBucketCheckpoints) { bb.putShort((short)vBucketCheckpoints.size()); for (Short vBucket : vBucketCheckpoints.keySet()) { bb.putShort(vBucket); bb.putLong(vBucketCheckpoints.get(vBucket)); } } return (ByteBuffer) bb.flip(); } }