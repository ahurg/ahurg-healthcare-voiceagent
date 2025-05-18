from dotenv import load_dotenv

from livekit import agents, api
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    function_tool,
    get_job_context,
    ChatContext,
    BackgroundAudioPlayer,
    AudioConfig,
    BuiltinAudioClip,
    RunContext
)
from livekit.plugins import (
    openai,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from state.user import UserInfo
load_dotenv()


class ConsentCollector(Agent):
    def __init__(self) -> None:
        stt=openai.STT(model="gpt-4o-mini-transcribe")
        llm=openai.LLM(model="gpt-4o-mini")
        tts=openai.TTS(model="gpt-4o-mini-tts", voice="ash")
        vad=silero.VAD.load()
        super().__init__(
            instructions="""Your are a voice AI agent with the singular task to collect positive 
            recording consent from the user. If consent is not given, you must end the call.""",
            llm=llm,
            stt=stt,
            tts=tts,
            vad=vad,
        )

    async def on_enter(self) -> None:
        await self.session.say("Welcome to ABC Health insurance member services?")

    @function_tool()
    async def on_consent_given(self):
        """Use this tool to indicate that consent has been given and the call may proceed."""

        # Perform a handoff, immediately transfering control to the new agent
        return HelpfulAssistant(chat_ctx=self.session._chat_ctx)

    @function_tool()
    async def end_call(self) -> None:
        """Use this tool to indicate that consent has not been given and the call should end."""
        await self.session.say("Thank you for your time, have a wonderful day.")
        job_ctx = get_job_context()
        print(f"Deleting room {job_ctx.room.name}")
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))


class HelpfulAssistant(Agent):
    def __init__(self, chat_ctx: ChatContext) -> None:
        stt=openai.STT(model="gpt-4o-mini-transcribe")
        llm=openai.LLM(model="gpt-4o-mini")
        tts=openai.TTS(model="gpt-4o-mini-tts", voice="sage")
        vad=silero.VAD.load()
        super().__init__(
            instructions="You are a helpful voice AI assistant specialized in Health care related queries. Your name is Bhanu.",
            chat_ctx=chat_ctx,
            llm=llm,
            stt=stt,
            tts=tts,
            vad=vad,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! My name is Bhanu specialized in Health care related queries. How can I help you today?")
        await self.session.say("Can you provide me your name?", allow_interruptions=False)

    @function_tool
    async def record_name(self, context: RunContext[UserInfo], name: str):
        """Use this tool to record the user's name."""
        context.userdata.user_name = name
        userdata: UserInfo = self.session.userdata
        await self.session.generate_reply(
            instructions=f"Greet {userdata.user_name} and ask them what health care related details they are looking for."
        )

    @function_tool
    async def get_claims(self):
        """Use this tool to provide claim related details."""
        userdata: UserInfo = self.session.userdata
        await self.session.generate_reply(
            instructions=f"Greet {userdata.user_name} and tell them they have 2 pending claims."
        )
    
    
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession[UserInfo](
        userdata=UserInfo(),
        # stt=openai.STT(model="gpt-4o-mini-transcribe"),
        # llm=openai.LLM(model="gpt-4o-mini"),
        # tts=openai.TTS(model="gpt-4o-mini-tts", voice="ash"),
        # vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=ConsentCollector(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    background_audio = BackgroundAudioPlayer(
        # play office ambience sound looping in the background
        # ambient_sound=AudioConfig(
        #     BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.8),
        # play keyboard typing sound when the agent is thinking
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
        ],
    )

    await background_audio.start(room=ctx.room, agent_session=session)

    

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
