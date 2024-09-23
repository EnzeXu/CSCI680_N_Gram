public class PortForwardBean extends AbstractBean { public static final String BEAN_NAME = "portforward" ; private long id = -1 ; private long hostId = -1 ; private String nickname = null ; private String type = null ; private int sourcePort = -1 ; private String destAddr = null ; private int destPort = -1 ; private boolean enabled = false ; private Object identifier = null ; public PortForwardBean ( long id , long hostId , String nickname , String type , int sourcePort , String destAddr , int destPort ) { this . id = id ; this . hostId = hostId ; this . nickname = nickname ; this . type = type ; this . sourcePort = sourcePort ; this . destAddr = destAddr ; this . destPort = destPort ; } public PortForwardBean ( long hostId , String nickname , String type , String source , String dest ) { this . hostId = hostId ; this . nickname = nickname ; this . type = type ; this . sourcePort = Integer . parseInt ( source ) ; setDest ( dest ) ; } @ Override public String getBeanName ( ) { return BEAN_NAME ; } public void setId ( long id ) { this . id = id ; } public long getId ( ) { return id ; } public void setNickname ( String nickname ) { this . nickname = nickname ; } public String getNickname ( ) { return nickname ; } public void setType ( String type ) { this . type = type ; } public String getType ( ) { return type ; } public void setSourcePort ( int sourcePort ) { this . sourcePort = sourcePort ; } public int getSourcePort ( ) { return sourcePort ; } public final void setDest ( String dest ) { String [ ] destSplit = dest . split ( " : " , -1 ) ; this . destAddr = destSplit [ 0 ] ; if ( destSplit . length > 1 ) { this . destPort = Integer . parseInt ( destSplit [ destSplit . length - 1 ] ) ; } } public void setDestAddr ( String destAddr ) { this . destAddr = destAddr ; } public String getDestAddr ( ) { return destAddr ; } public void setDestPort ( int destPort ) { this . destPort = destPort ; } public int getDestPort ( ) { return destPort ; } public void setEnabled ( boolean enabled ) { this . enabled = enabled ; } public boolean isEnabled ( ) { return enabled ; } public void setIdentifier ( Object identifier ) { this . identifier = identifier ; } public Object getIdentifier ( ) { return identifier ; } @ SuppressLint ( "DefaultLocale" ) public CharSequence getDescription ( ) { String description = "Unknown type" ; if ( HostDatabase . PORTFORWARD_LOCAL . equals ( type ) ) { description = String . format ( "Local port %d to %s : %d" , sourcePort , destAddr , destPort ) ; } else if ( HostDatabase . PORTFORWARD_REMOTE . equals ( type ) ) { description = String . format ( "Remote port %d to %s : %d" , sourcePort , destAddr , destPort ) ; } else if ( HostDatabase . PORTFORWARD_DYNAMIC5 . equals ( type ) ) { description = String . format ( "Dynamic port %d ( SOCKS ) " , sourcePort ) ; } return description ; } @ Override public ContentValues getValues ( ) { ContentValues values = new ContentValues ( ) ; values . put ( HostDatabase . FIELD_PORTFORWARD_HOSTID , hostId ) ; values . put ( HostDatabase . FIELD_PORTFORWARD_NICKNAME , nickname ) ; values . put ( HostDatabase . FIELD_PORTFORWARD_TYPE , type ) ; values . put ( HostDatabase . FIELD_PORTFORWARD_SOURCEPORT , sourcePort ) ; values . put ( HostDatabase . FIELD_PORTFORWARD_DESTADDR , destAddr ) ; values . put ( HostDatabase . FIELD_PORTFORWARD_DESTPORT , destPort ) ; return values ; } }