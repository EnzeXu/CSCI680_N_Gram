public class PwDate implements Cloneable { private static final int DATE_SIZE = 5; private boolean cDateBuilt = false; private boolean jDateBuilt = false; private Date jDate; private byte[] cDate; public PwDate(byte[] buf, int offset) { cDate = new byte[DATE_SIZE]; System.arraycopy(buf, offset, cDate, 0, DATE_SIZE); cDateBuilt = true; } public PwDate(Date date) { jDate = date; jDateBuilt = true; } public PwDate(long millis) { jDate = new Date(millis); jDateBuilt = true; } private PwDate() { } @Override public Object clone() { PwDate copy = new PwDate(); if ( cDateBuilt ) { byte[] newC = new byte[DATE_SIZE]; System.arraycopy(cDate, 0, newC, 0, DATE_SIZE); copy.cDate = newC; copy.cDateBuilt = true; } if ( jDateBuilt ) { copy.jDate = (Date) jDate.clone(); copy.jDateBuilt = true; } return copy; } public Date getJDate() { if ( ! jDateBuilt ) { jDate = readTime(cDate, 0, App.getCalendar()); jDateBuilt = true; } return jDate; } public byte[] getCDate() { if ( ! cDateBuilt ) { cDate = writeTime(jDate, App.getCalendar()); cDateBuilt = true; } return cDate; } public static Date readTime(byte[] buf, int offset, Calendar time) { int dw1 = Types.readUByte(buf, offset); int dw2 = Types.readUByte(buf, offset + 1); int dw3 = Types.readUByte(buf, offset + 2); int dw4 = Types.readUByte(buf, offset + 3); int dw5 = Types.readUByte(buf, offset + 4); int year = (dw1 << 6) | (dw2 >> 2); int month = ((dw2 & 0x00000003) << 2) | (dw3 >> 6); int day = (dw3 >> 1) & 0x0000001F; int hour = ((dw3 & 0x00000001) << 4) | (dw4 >> 4); int minute = ((dw4 & 0x0000000F) << 2) | (dw5 >> 6); int second = dw5 & 0x0000003F; if (time == null) { time = Calendar.getInstance(); } time.set(year, month - 1, day, hour, minute, second); return time.getTime(); } public static byte[] writeTime(Date date) { return writeTime(date, null); } public static byte[] writeTime(Date date, Calendar cal) { if (date == null) { return null; } byte[] buf = new byte[5]; if (cal == null) { cal = Calendar.getInstance(); } cal.setTime(date); int year = cal.get(Calendar.YEAR); int month = cal.get(Calendar.MONTH) + 1; int day = cal.get(Calendar.DAY_OF_MONTH) - 1; int hour = cal.get(Calendar.HOUR_OF_DAY); int minute = cal.get(Calendar.MINUTE); int second = cal.get(Calendar.SECOND); buf[0] = Types.writeUByte(((year >> 6) & 0x0000003F)); buf[1] = Types.writeUByte(((year & 0x0000003F) << 2) | ((month >> 2) & 0x00000003)); buf[2] = (byte) (((month & 0x00000003) << 6) | ((day & 0x0000001F) << 1) | ((hour >> 4) & 0x00000001)); buf[3] = (byte) (((hour & 0x0000000F) << 4) | ((minute >> 2) & 0x0000000F)); buf[4] = (byte) (((minute & 0x00000003) << 6) | (second & 0x0000003F)); return buf; } @Override public boolean equals(Object o) { if ( this == o ) { return true; } if ( o == null ) { return false; } if ( getClass() != o.getClass() ) { return false; } PwDate date = (PwDate) o; if ( cDateBuilt && date.cDateBuilt ) { return Arrays.equals(cDate, date.cDate); } else if ( jDateBuilt && date.jDateBuilt ) { return IsSameDate(jDate, date.jDate); } else if ( cDateBuilt && date.jDateBuilt ) { return Arrays.equals(date.getCDate(), cDate); } else { return IsSameDate(date.getJDate(), jDate); } } public static boolean IsSameDate(Date d1, Date d2) { Calendar cal1 = Calendar.getInstance(); cal1.setTime(d1); cal1.set(Calendar.MILLISECOND, 0); Calendar cal2 = Calendar.getInstance(); cal2.setTime(d2); cal2.set(Calendar.MILLISECOND, 0); return (cal1.get(Calendar.YEAR) == cal2.get(Calendar.YEAR)) && (cal1.get(Calendar.MONTH) == cal2.get(Calendar.MONTH)) && (cal1.get(Calendar.DAY_OF_MONTH) == cal2.get(Calendar.DAY_OF_MONTH)) && (cal1.get(Calendar.HOUR) == cal2.get(Calendar.HOUR)) && (cal1.get(Calendar.MINUTE) == cal2.get(Calendar.MINUTE)) && (cal1.get(Calendar.SECOND) == cal2.get(Calendar.SECOND)); } }