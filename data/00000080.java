public class MessageBuilder { private RequestMessage message; public MessageBuilder() { this.message = new RequestMessage(); message.setMagic(TapMagic.PROTOCOL_BINARY_REQ); message.setOpcode(TapOpcode.REQUEST); } public void doBackfill(long date) { message.setBackfill(date); message.setFlags(TapRequestFlag.BACKFILL); } public void doDump() { message.setFlags(TapRequestFlag.DUMP); } public void specifyVbuckets(short[] vbucketlist) { message.setVbucketlist(vbucketlist); message.setFlags(TapRequestFlag.LIST_VBUCKETS); } public void supportAck() { message.setFlags(TapRequestFlag.SUPPORT_ACK); } public void keysOnly() { message.setFlags(TapRequestFlag.KEYS_ONLY); } public void takeoverVbuckets(short[] vbucketlist) { message.setVbucketlist(vbucketlist); message.setFlags(TapRequestFlag.TAKEOVER_VBUCKETS); } public RequestMessage getMessage() { return message; } }