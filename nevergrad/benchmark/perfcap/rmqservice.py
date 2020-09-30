#################################################################
# Author: Alex Doumanoglou (al3x.doum@gmail.com / aldoum@iti.gr)
# Information Technologies Institute - Visual Computing Lab (https://vcl.iti.gr)
# 30 Sept 2020
#################################################################

import asyncio
import typing as tp
import threading
import logging
import json
from uuid import uuid4 as uuid_gen
from concurrent.futures import Future
import aio_pika


class RMQSettings:
    def __init__(self, conn_str: str, exchange_in: str, exchange_out: str, routing_key_in: str = None):
        self.connection_string: str = conn_str
        self.exchange_in: str = exchange_in
        self.exchange_out: str = exchange_out
        self.routing_key_in: str = routing_key_in


class RMQService:
    """
    RabbitMQ Client, based on asyncio and aio_pika
    This network module is responsible for low-level communication with the Perfcap Benchmark Server
    This is a threaded implemenation. Network I/O takes place in a background thread
    All requests to the benchmark server happen through the request method, which appears to the caller as a blocking call until
    the request has been served by the Perfcap Benchmark Server and a reply has been received.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, rmq_settings: RMQSettings, do_rmq_exch_cleanup: bool = False):

        self._rmq_settings = rmq_settings
        self._event_loop: asyncio.AbstractEventLoop = None
        self._thread: threading.Thread = None
        self._request_futures: tp.Dict[str, Future] = {}
        self._consumer_tag: aio_pika.queue.ConsumerTag = None

        self._connection: aio_pika.ConnectionType = None
        self._channel: aio_pika.Channel = None
        self._exch_in: aio_pika.Exchange = None
        self._exch_out: aio_pika.Exchange = None
        self._queue_in: aio_pika.Queue = None
        self._pending_futures: tp.Set[Future] = set()
        self._do_rmq_exch_cleanup: bool = do_rmq_exch_cleanup
        self._logger = logging.getLogger("RMQService")

    def _shutdown_futures(self):
        """
        Future clean up on shutdown
        """
        for (_, ft) in self._request_futures.items():
            ft.set_exception(Exception("RMQService Future aborted."))
        for ft in self._pending_futures:
            ft.set_exception(Exception("RMQService Future aborted."))
        self._request_futures.clear()
        self._pending_futures.clear()

    async def _process_message(self, message: aio_pika.IncomingMessage) -> tp.Dict[str, tp.Any]:
        """
        Processes incoming message from RMQ
        """
        async with message.process():
            json_msg = json.loads(message.body)
            tid = json_msg["reply_info"]["transaction_id"]          # get transaction_id from the reply message
            ft = self._request_futures.pop(tid, None)               # obtain future from dictionary
            if ft is not None:
                ft.set_result(json_msg["args"])                     # set future result from received message

    async def _setup(self, loop: asyncio.AbstractEventLoop):
        """
        Connect to RMQ Server and setup Exchanges/Queues.
        """
        try:

            self._connection = await aio_pika.connect(self._rmq_settings.connection_string, loop=loop)
            self._channel = await self._connection.channel()
            self._exch_in = await self._channel.declare_exchange(self._rmq_settings.exchange_in, type=aio_pika.ExchangeType.FANOUT,
                                                                 auto_delete=False)
            self._exch_out = await self._channel.declare_exchange(self._rmq_settings.exchange_out, type=aio_pika.ExchangeType.FANOUT,
                                                                  auto_delete=True)

            self._queue_in = await self._channel.declare_queue("", auto_delete=True)

            if self._rmq_settings.routing_key_in is None:
                await self._queue_in.bind(self._exch_in)
            else:
                await self._queue_in.bind(self._exch_in, self._rmq_settings.routing_key_in)
            self._consumer_tag = await self._queue_in.consume(self._process_message)
        except aio_pika.AMQPException as ex:
            self._logger.error("RMQService setup failed... shutting down.")
            self._logger.exception(ex)
            self._event_loop.stop()

    async def _shutdown(self):
        """
        Clean up function when RMQService is stopped
        """
        if self._queue_in is not None:
            await self._queue_in.cancel(self._consumer_tag)
            await self._queue_in.delete()

        if self._do_rmq_exch_cleanup:
            if self._exch_in is not None:
                await self._exch_in.delete()
            if self._exch_out is not None:
                await self._exch_out.delete()

        if self._channel is not None:
            await self._channel.close()
        if self._connection is not None:
            await self._connection.close()

        self._queue_in = None
        self._exch_in = None
        self._exch_out = None
        self._channel = None
        self._connection = None

        self._shutdown_futures()

    def _main(self):
        """
        Main method of background thread
        """
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.run_forever()
        self._event_loop.run_until_complete(self._shutdown())
        self._event_loop.close()

    async def _request(self, msg: tp.Dict[str, tp.Any], ft: Future):
        """
        Core function to submit requests to Perfcap Benchmark Server through RMQ.
        """
        self._pending_futures.remove(ft)
        tid = str(uuid_gen())               # generate a transaction_id (i.e. request id)
        self._request_futures[tid] = ft     # add future to _request_future dictionary under transaction_id key
        request_info: tp.Dict[str, str] = {
            # embed listener's routing key to the message. benchmark server will reply to this request in
            # the routing_key specified in reply_id
            "reply_id": self._rmq_settings.routing_key_in if self._rmq_settings.routing_key_in is not None else "",
            # mark transaction with id. this will help identify the reply of the server.
            # The server always embeds transaction_id to its reply message.
            "transaction_id": tid
        }
        fmsg = {}
        fmsg["request_info"] = request_info
        fmsg["args"] = msg
        await self._exch_out.publish(aio_pika.Message(bytes(json.dumps(fmsg), encoding='utf-8')), routing_key="")

    def _request_safe(self, msg: tp.Dict[str, tp.Any], ft: Future):
        self._pending_futures.add(ft)
        if self._exch_out is not None:  # if nevegrad output exchange has been created enqueue request
            asyncio.create_task(self._request(msg, ft))
        else:
            # exchange is not ready yet (we are still connecting to RMQ Server). Requeue, for later
            self._event_loop.call_later(0.3, self._request_safe, msg, ft)

    def request(self, msg: tp.Dict[str, tp.Any]) -> tp.Union[None, tp.Dict[str, tp.Any]]:
        """
        Submit a request to the Perfcap Benchmark Server.
        This is a blocking call returning the result replied from the server.
        You must first call run() for this method to have any effect.
        """
        if self._thread is not None:
            ft = Future()
            self._event_loop.call_soon_threadsafe(self._request_safe, msg, ft)
            return ft.result()
        return None

    def request_forced(self, msg: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
        """
        Submit a request to the Perfcap Benchmark Server.
        This is a blocking call returning the result replied from the server
        In case the RMQService is not in running state, it is silently started
        """
        if self._thread is None:
            self.run()

        ft = Future()
        self._event_loop.call_soon_threadsafe(self._request_safe, msg, ft)
        return ft.result()

    def run(self):
        """
        This method initiates network connection to the RMQ Server in a background thread.
        """
        if self._thread is not None:
            return

        self._event_loop = asyncio.new_event_loop()
        self._event_loop.call_soon_threadsafe(lambda: asyncio.create_task(self._setup(self._event_loop)))
        self._thread = threading.Thread(target=self._main)
        self._thread.start()

    def stop(self):
        """
        Call this method to disconnect from RMQServer and stop the background thread.
        """
        if (self._event_loop is not None) and (self._thread is not None):
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            self._thread.join()
            self._thread = None
            self._event_loop = None
