public class PwIconCustom extends PwIcon { public static final PwIconCustom ZERO = new PwIconCustom ( PwDatabaseV4 . UUID_ZERO , new byte [ 0 ] ) ; public final UUID uuid ; public byte [ ] imageData ; public Date lastMod = null ; public String name = "" ; public PwIconCustom ( UUID u , byte [ ] data ) { uuid = u ; imageData = data ; } @ Override public int hashCode ( ) { final int prime = 31 ; int result = 1 ; result = prime * result + ( ( uuid == null ) ? 0 : uuid . hashCode ( ) ) ; return result ; } @ Override public boolean equals ( Object obj ) { if ( this == obj ) return true ; if ( obj == null ) return false ; if ( getClass ( ) != obj . getClass ( ) ) return false ; PwIconCustom other = ( PwIconCustom ) obj ; if ( uuid == null ) { if ( other . uuid != null ) return false ; } else if ( !uuid . equals ( other . uuid ) ) return false ; return true ; } }